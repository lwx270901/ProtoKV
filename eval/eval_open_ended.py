from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

try:
    import openai
except Exception:  # pragma: no cover
    openai = None


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT evaluation for open-ended StreamingBench/RVS outputs from qwen_inference_online.py"
    )
    parser.add_argument("--pred_path", required=True, help="CSV produced by qwen_inference_online.py: video_id,question,answer,pred_answer,answer_type,delta,ttft,e2e")
    parser.add_argument("--output_dir", required=True, help="Directory for per-sample annotation JSON files")
    parser.add_argument("--output_json", required=True, help="Path for final combined JSON output")
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional per-sample CSV in the requested format: Method,Question_id,Answer_type,delta,Correction,score,TTFT,E2Elatency",
    )
    parser.add_argument("--method_name", default="ProtoKV", help="Value to write in the Method column of --output_csv")
    parser.add_argument("--source_data_path", default=None, help="Optional original StreamingBench/RVS JSON. Used to reconstruct exact Question_id as <video_id>_<conversation_index>.")
    parser.add_argument("--question_id_start", default=0, type=int, help="Fallback per-video question index start when Question_id cannot be reconstructed from --source_data_path")
    parser.add_argument("--summary_csv", default=None, help="Optional aggregate CSV containing overall/per-delta summary")
    parser.add_argument("--num_tasks", default=8, type=int, help="Number of parallel worker processes")

    # The evaluator is for open-ended questions. It does not filter by answer type;
    # the Answer_type column is copied from the input file into the output CSV.
    parser.add_argument(
        "--delta_filter",
        default="all",
        help="Comma-separated delta values to evaluate, e.g. '0' or '0,10'. Use 'all' to evaluate every delta.",
    )

    # OpenAI-compatible API settings.
    parser.add_argument("--model_name", default="gpt-3.5-turbo-0613")
    parser.add_argument("--base_url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--max_tokens", default=300, type=int)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--retry", default=10, type=int)
    parser.add_argument("--sleep", default=1.0, type=float)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-evaluate rows even if their annotation JSON already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Prepare/normalize predictions and write a preview JSON, but do not call GPT.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _mkdir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _norm_answer_type(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"_+", "_", s)
    return s


def _parse_filter_values(spec: str) -> Optional[set]:
    if spec is None:
        return None
    spec = str(spec).strip()
    if not spec or spec.lower() == "all":
        return None
    return {_norm_answer_type(x) for x in spec.split(",") if str(x).strip()}


def _parse_delta_filter(spec: str) -> Optional[set]:
    if spec is None:
        return None
    spec = str(spec).strip()
    if not spec or spec.lower() == "all":
        return None
    out = set()
    for x in spec.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.add(str(float(x)))
        except Exception:
            out.add(x)
    return out


def _delta_key(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "standard"
    s = str(x).strip()
    if not s:
        return "standard"
    try:
        return str(float(s))
    except Exception:
        return s


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    return str(x)


def _norm_match_text(x: Any) -> str:
    s = _safe_text(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def shrink_repeated_words(text: str) -> str:
    """Light cleanup for repetitive generated answers."""
    text = _safe_text(text).strip()
    if not text:
        return ""
    parts = text.split()
    out: List[str] = []
    for part in parts:
        if not out or part != out[-1]:
            out.append(part)
        else:
            # Stop at immediate runaway repetition, matching the original script's intent.
            break
    return " ".join(out)


def _short_hash(text: str, n: int = 10) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def _get_first_existing(row: Dict[str, Any], names: Sequence[str], default: str = "") -> str:
    for name in names:
        if name in row:
            val = _safe_text(row.get(name))
            if val != "":
                return val
    return default


def build_question_id_maps(source_data_path: Optional[str]) -> Tuple[Dict[Tuple[str, str, str, str], str], Dict[Tuple[str, str], str]]:
    """Map original StreamingBench/RVS conversations to benchmark Question_id values.

    If the source JSON is provided, question ids are reconstructed as
    <video_id>_<conversation_index>, unless a conversation already has a
    question_id/Question_id field. This is the format used by the reference CSV.
    """
    exact: Dict[Tuple[str, str, str, str], str] = {}
    qonly: Dict[Tuple[str, str], str] = {}
    if not source_data_path:
        return exact, qonly
    if not os.path.exists(source_data_path):
        print(f"Warning: --source_data_path not found: {source_data_path}. Falling back to generated Question_id values.")
        return exact, qonly
    with open(source_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for candidate_key in ("data", "videos", "samples", "annotations"):
            if isinstance(data.get(candidate_key), list):
                data = data[candidate_key]
                break
    if not isinstance(data, list):
        print("Warning: --source_data_path is not a list-style JSON. Falling back to generated Question_id values.")
        return exact, qonly
    for item in data:
        if not isinstance(item, dict):
            continue
        vid = _safe_text(item.get("video_id", item.get("id", "")))
        conversations = item.get("conversations", []) or item.get("qa", []) or item.get("questions", []) or []
        if not isinstance(conversations, list):
            continue
        for j, conv in enumerate(conversations):
            if not isinstance(conv, dict):
                continue
            q = conv.get("question", conv.get("query", conv.get("prompt", "")))
            a = conv.get("answer", conv.get("gt_answer", conv.get("ground_truth", "")))
            at = conv.get("answer_type", conv.get("Answer_type", ""))
            qid = _safe_text(conv.get("question_id", conv.get("Question_id", "")))
            if not qid:
                qid = f"{vid}_{j}"
            exact[(vid, _norm_match_text(q), _norm_match_text(a), _norm_answer_type(at))] = qid
            qonly[(vid, _norm_match_text(q))] = qid
    print(f"Loaded {len(exact)} question_id mappings from {source_data_path}")
    return exact, qonly


@dataclass
class EvalItem:
    key: str
    row_index: int
    video_id: str
    question_id: str
    question: str
    answer: str
    pred_answer: str
    answer_type: str
    delta: str
    extra: Dict[str, Any]

    def to_qa_set(self) -> Dict[str, Any]:
        return {
            "row_index": self.row_index,
            "video_id": self.video_id,
            "question_id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "pred_answer": self.pred_answer,
            "answer_type": self.answer_type,
            "delta": self.delta,
            **self.extra,
        }


def load_prediction_items(
    pred_path: str,
    delta_filter: Optional[set],
    qid_exact: Optional[Dict[Tuple[str, str, str, str], str]] = None,
    qid_qonly: Optional[Dict[Tuple[str, str], str]] = None,
    question_id_start: int = 0,
) -> List[EvalItem]:
    df = pd.read_csv(pred_path)
    expected_reference_cols = {"video_id", "question", "answer", "pred_answer", "answer_type", "delta", "ttft", "e2e"}
    missing_reference_cols = expected_reference_cols.difference(set(df.columns))
    if missing_reference_cols:
        print(
            "Warning: prediction CSV is not exactly the reference schema; "
            f"missing columns: {sorted(missing_reference_cols)}. "
            "The evaluator will try compatible aliases where possible."
        )
    if len(df) == 0:
        raise ValueError(f"Empty prediction file: {pred_path}")

    rows = df.to_dict(orient="records")
    items: List[EvalItem] = []
    skipped_delta = 0
    skipped_empty = 0
    qid_exact = qid_exact or {}
    qid_qonly = qid_qonly or {}
    # Stable fallback Question_id when inference CSV does not include one and no source mapping is available.
    # The same (video_id, question, answer) keeps the same id across deltas.
    fallback_qid_map: Dict[Tuple[str, str, str], str] = {}
    per_video_counter: Dict[str, int] = {}

    for idx, row in enumerate(rows):
        video_id = _get_first_existing(row, ["video_id", "video", "id"], default=f"row{idx}")
        question = _get_first_existing(row, ["question", "query", "prompt"])
        answer = _get_first_existing(row, ["answer", "gt_answer", "ground_truth", "label"])
        pred = _get_first_existing(row, ["pred_answer", "prediction", "pred", "output", "response"])
        pred = shrink_repeated_words(pred)
        question_id = _get_first_existing(
            row,
            ["Question_id", "question_id", "qid", "sample_id", "item_id", "id"],
            default="",
        )
        answer_type_raw = _get_first_existing(row, ["Answer_type", "answer_type", "answerType", "type"], default="")
        answer_type_norm = _norm_answer_type(answer_type_raw)
        if not question_id:
            exact_key = (video_id, _norm_match_text(question), _norm_match_text(answer), answer_type_norm)
            question_id = qid_exact.get(exact_key, "")
        if not question_id:
            question_id = qid_qonly.get((video_id, _norm_match_text(question)), "")
        if not question_id:
            qid_key = (video_id, _norm_match_text(question), _norm_match_text(answer))
            if qid_key not in fallback_qid_map:
                next_idx = per_video_counter.get(video_id, int(question_id_start))
                per_video_counter[video_id] = next_idx + 1
                fallback_qid_map[qid_key] = f"{video_id}_{next_idx}"
            question_id = fallback_qid_map[qid_key]
        delta = _delta_key(row.get("delta", "standard"))

        if delta_filter is not None:
            if delta not in delta_filter:
                skipped_delta += 1
                continue

        if not question or not answer:
            skipped_empty += 1
            continue

        # Keep timing and any other columns in the annotation for later analysis.
        extra = {}
        for k, v in row.items():
            if k not in {
                "video_id", "question", "answer", "pred_answer", "answer_type", "Answer_type", "answerType", "type", "delta",
                "Question_id", "question_id", "qid", "sample_id", "item_id", "id",
            }:
                extra[k] = _safe_text(v)

        # Use row index to guarantee uniqueness even when video_id/question/delta repeat.
        key_base = f"{question_id}__delta_{delta}__row_{idx:06d}"
        # Avoid path separators or weird chars.
        key = re.sub(r"[^A-Za-z0-9_.=-]+", "_", key_base)
        items.append(EvalItem(
            key=key,
            row_index=idx,
            video_id=video_id,
            question_id=question_id,
            question=question,
            answer=answer,
            pred_answer=pred,
            answer_type=answer_type_raw,
            delta=delta,
            extra=extra,
        ))

    print(f"Loaded {len(rows)} prediction rows from {pred_path}")
    print(f"Kept {len(items)} rows for evaluation")
    if skipped_delta:
        print(f"Skipped by delta_filter: {skipped_delta}")
    if skipped_empty:
        print(f"Skipped rows missing question/answer: {skipped_empty}")
    return items


# -----------------------------------------------------------------------------
# GPT evaluation
# -----------------------------------------------------------------------------

class GPTService:
    def __init__(self, model_name: str, base_url: str, api_key: str, max_tokens: int, temperature: float, retry: int, sleep: float):
        if openai is None:
            raise RuntimeError("The openai package is not installed. Install it or run with --dry_run.")
        if not api_key:
            raise RuntimeError("Missing API key. Set OPENAI_API_KEY or pass --api_key.")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retry = retry
        self.sleep = sleep
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)

    def _gpt_response(self, messages: List[Dict[str, str]]) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return completion.choices[0].message.content or ""

    def gpt_with_retry(self, messages: List[Dict[str, str]]) -> Optional[str]:
        last_error = None
        for _ in range(self.retry):
            try:
                result = self._gpt_response(messages)
                if result:
                    return result
            except Exception as e:
                last_error = e
                print(f"An error occurred: {e}")
            time.sleep(self.sleep)
        print(f"GPT failed after {self.retry} retries. Last error: {last_error}")
        return None


def make_eval_prompt(item: EvalItem) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an evaluator for open-ended video question answering. "
                "Compare the model's predicted answer with the reference answer. "
                "Give credit for semantic equivalence, synonyms, paraphrases, and answers that are correct even if wording differs. "
                "Penalize hallucinated, contradictory, irrelevant, or missing answers."
            ),
        },
        {
            "role": "user",
            "content": (
                "Evaluate the following video-based question-answer pair.\n\n"
                f"Question: {item.question}\n"
                f"Correct Answer: {item.answer}\n"
                f"Predicted Answer: {item.pred_answer}\n\n"
                "Return only a Python dictionary string with keys 'pred' and 'score'. "
                "'pred' must be 'yes' if the predicted answer meaningfully matches the correct answer, otherwise 'no'. "
                "'score' must be a numeric value from 0 to 5, where 5 is perfect. "
                "Do not include any explanation. Example: {'pred': 'yes', 'score': 4}"
            ),
        },
    ]


def parse_gpt_eval_response(text: Optional[str]) -> Dict[str, Any]:
    if not text:
        return {"pred": "error", "score": 0, "raw": text or ""}
    raw = text.strip()
    # Try direct Python dict parsing first.
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Try extracting {...} from accidental extra text.
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            obj = ast.literal_eval(m.group(0))
            if isinstance(obj, dict):
                obj["raw"] = raw
                return obj
        except Exception:
            pass
    return {"pred": "error", "score": 0, "raw": raw}


def annotate_worker(
    items_by_key: Dict[str, Dict[str, Any]],
    keys: Sequence[str],
    output_dir: str,
    args_dict: Dict[str, Any],
) -> None:
    service = None
    if not args_dict.get("dry_run", False):
        service = GPTService(
            model_name=args_dict["model_name"],
            base_url=args_dict.get("base_url", ""),
            api_key=args_dict.get("api_key", ""),
            max_tokens=int(args_dict.get("max_tokens", 300)),
            temperature=float(args_dict.get("temperature", 0.0)),
            retry=int(args_dict.get("retry", 10)),
            sleep=float(args_dict.get("sleep", 1.0)),
        )

    for key in tqdm(keys):
        item_dict = items_by_key[key]
        item = EvalItem(**item_dict)
        out_path = os.path.join(output_dir, f"{key}.json")
        try:
            if args_dict.get("dry_run", False):
                result = [{"pred": "dry_run", "score": 0}, item.to_qa_set()]
            else:
                messages = make_eval_prompt(item)
                response_text = service.gpt_with_retry(messages) if service else None
                response_dict = parse_gpt_eval_response(response_text)
                result = [response_dict, item.to_qa_set()]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # Write an error file to avoid infinite re-processing loops.
            result = [{"pred": "error", "score": 0, "error": str(e)}, item.to_qa_set()]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Error processing key={key}: {e}")


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

def _to_score(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0



def _format_score(x: Any) -> str:
    try:
        val = float(x)
        if math.isnan(val):
            return ""
        return f"{val:.4f}"
    except Exception:
        return ""


def _format_delta(x: Any) -> str:
    key = _delta_key(x)
    if key == "standard":
        return "0.0000"
    try:
        val = float(key)
        if math.isnan(val):
            return ""
        return f"{val:.4f}"
    except Exception:
        return key


def _format_latency_seconds(sample: Dict[str, Any], names_ms: Sequence[str], names_seconds: Sequence[str]) -> str:
    # Current qwen_inference_online.py writes ttft/e2e in seconds, matching
    # the reference CSV. For backward compatibility, *_ms fields are divided
    # by 1000 if they appear in older prediction files.
    for name in names_ms:
        raw = sample.get(name, "")
        try:
            val = float(raw)
            if math.isnan(val):
                continue
            return f"{val / 1000.0:.4f}"
        except Exception:
            pass
    for name in names_seconds:
        raw = sample.get(name, "")
        try:
            val = float(raw)
            if math.isnan(val):
                continue
            return f"{val:.4f}"
        except Exception:
            pass
    return ""


def write_requested_format_csv(path: str, combined: Dict[str, Any], method_name: str) -> None:
    """Write per-sample evaluation output in the exact requested schema.

    Columns:
        Method, Question_id, Answer_type, delta, Correction, score, TTFT, E2Elatency
    """
    records: List[Dict[str, Any]] = []
    for key, result in combined.items():
        if not isinstance(result, list) or len(result) < 2:
            continue
        ev = result[0] if isinstance(result[0], dict) else {}
        sample = result[1] if isinstance(result[1], dict) else {}
        pred_label = _safe_text(ev.get("pred", ev.get("prev", ""))).strip().lower()
        if "yes" in pred_label:
            correction = "Y"
        elif "no" in pred_label:
            correction = "N"
        else:
            correction = "N"
        records.append({
            "Method": method_name,
            "Question_id": _safe_text(sample.get("question_id", sample.get("video_id", key))),
            "Answer_type": _safe_text(sample.get("answer_type", "")),
            "delta": _format_delta(sample.get("delta", "")),
            "Correction": correction,
            "score": _format_score(ev.get("score", 0)),
            "TTFT": _format_latency_seconds(
                sample,
                names_ms=["ttft_ms", "TTFT_ms", "ttft_milliseconds"],
                names_seconds=["TTFT", "ttft", "ttft_s"],
            ),
            "E2Elatency": _format_latency_seconds(
                sample,
                names_ms=["e2e_ms", "E2Elatency_ms", "e2e_milliseconds"],
                names_seconds=["E2Elatency", "E2E", "e2e", "e2e_seconds"],
            ),
        })
    columns = ["Method", "Question_id", "Answer_type", "delta", "Correction", "score", "TTFT", "E2Elatency"]

    def _sort_key(rec: Dict[str, Any]) -> Tuple[str, float]:
        try:
            d = float(rec.get("delta", 0.0))
        except Exception:
            d = 0.0
        return (_safe_text(rec.get("Question_id", "")), d)

    records.sort(key=_sort_key)
    pd.DataFrame(records, columns=columns).to_csv(path, index=False)

def summarize(combined: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for key, result in combined.items():
        if not isinstance(result, list) or len(result) < 2:
            continue
        ev = result[0] if isinstance(result[0], dict) else {}
        sample = result[1] if isinstance(result[1], dict) else {}
        pred_label = _safe_text(ev.get("pred", ev.get("prev", ""))).lower()
        score = _to_score(ev.get("score", 0))
        is_yes = "yes" in pred_label
        is_no = "no" in pred_label
        rows.append({
            "key": key,
            "delta": _delta_key(sample.get("delta", "standard")),
            "answer_type": _safe_text(sample.get("answer_type", "")),
            "score": score,
            "yes": int(is_yes),
            "no": int(is_no),
            "valid_yes_no": int(is_yes or is_no),
        })

    def stats(subrows: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(subrows)
        valid = sum(r["valid_yes_no"] for r in subrows)
        yes = sum(r["yes"] for r in subrows)
        no = sum(r["no"] for r in subrows)
        score_sum = sum(r["score"] for r in subrows)
        return {
            "count": n,
            "valid_yes_no": valid,
            "yes_count": yes,
            "no_count": no,
            "accuracy": (yes / valid) if valid else 0.0,
            "average_score": (score_sum / n) if n else 0.0,
        }

    overall = stats(rows)
    by_delta: Dict[str, Any] = {}
    for delta in sorted({r["delta"] for r in rows}):
        by_delta[delta] = stats([r for r in rows if r["delta"] == delta])
    return {"overall": overall, "by_delta": by_delta}


def write_summary_csv(path: str, summary: Dict[str, Any]) -> None:
    records: List[Dict[str, Any]] = []
    overall = dict(summary.get("overall", {}))
    overall["group"] = "overall"
    overall["delta"] = "all"
    records.append(overall)
    for delta, stats in summary.get("by_delta", {}).items():
        rec = dict(stats)
        rec["group"] = "delta"
        rec["delta"] = delta
        records.append(rec)
    pd.DataFrame(records).to_csv(path, index=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _mkdir(args.output_dir)
    _mkdir(os.path.dirname(args.output_json) or ".")
    if args.output_csv:
        _mkdir(os.path.dirname(args.output_csv) or ".")
    if args.summary_csv:
        _mkdir(os.path.dirname(args.summary_csv) or ".")

    delta_filter = _parse_delta_filter(args.delta_filter)
    qid_exact, qid_qonly = build_question_id_maps(args.source_data_path)
    items = load_prediction_items(
        args.pred_path,
        delta_filter,
        qid_exact=qid_exact,
        qid_qonly=qid_qonly,
        question_id_start=args.question_id_start,
    )

    # Store by key for multiprocessing.
    items_by_key: Dict[str, Dict[str, Any]] = {item.key: item.__dict__ for item in items}
    all_keys = [item.key for item in items]

    if args.overwrite:
        pending_keys = all_keys
    else:
        completed = {os.path.splitext(x)[0] for x in os.listdir(args.output_dir) if x.endswith(".json")}
        pending_keys = [k for k in all_keys if k not in completed]

    print(f"Existing annotations: {len(all_keys) - len(pending_keys)}")
    print(f"Pending annotations: {len(pending_keys)}")

    if pending_keys:
        args_dict = vars(args).copy()
        if args.num_tasks <= 1 or len(pending_keys) <= 1:
            annotate_worker(items_by_key, pending_keys, args.output_dir, args_dict)
        else:
            num_tasks = max(1, min(int(args.num_tasks), len(pending_keys)))
            chunks = [pending_keys[i::num_tasks] for i in range(num_tasks)]
            chunks = [c for c in chunks if c]
            task_args = [(items_by_key, chunk, args.output_dir, args_dict) for chunk in chunks]
            with Pool(processes=len(chunks)) as pool:
                pool.starmap(annotate_worker, task_args)

    combined: Dict[str, Any] = {}
    for item in items:
        path = os.path.join(args.output_dir, f"{item.key}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                combined[item.key] = json.load(f)

    summary = summarize(combined)
    output = {
        "metadata": {
            "pred_path": args.pred_path,
            "num_prediction_rows_evaluated": len(items),
            "delta_filter": args.delta_filter,
            "evaluation_scope": "open_ended",
            "model_name": args.model_name,
            "method_name": args.method_name,
            "source_data_path": args.source_data_path,
        },
        "results": combined,
        "summary": summary,
        # Backward-compatible top-level fields used by the original script.
        "average_score": summary["overall"]["average_score"],
        "accuracy": summary["overall"]["accuracy"],
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if args.output_csv:
        write_requested_format_csv(args.output_csv, combined, args.method_name)

    if args.summary_csv:
        write_summary_csv(args.summary_csv, summary)

    print("All evaluation completed!")
    if args.output_csv:
        print(f"Requested-format CSV written to: {args.output_csv}")
    print(f"Evaluated rows: {summary['overall']['count']}")
    print(f"Yes count: {summary['overall']['yes_count']}")
    print(f"No count: {summary['overall']['no_count']}")
    print(f"Accuracy: {summary['overall']['accuracy'] * 100:.1f}%")
    print(f"Average score: {summary['overall']['average_score']:.2f}")
    if summary.get("by_delta"):
        print("By delta:")
        for delta, stats in summary["by_delta"].items():
            print(
                f"  delta={delta}: n={stats['count']} "
                f"acc={stats['accuracy'] * 100:.1f}% avg_score={stats['average_score']:.2f}"
            )


if __name__ == "__main__":
    main()
