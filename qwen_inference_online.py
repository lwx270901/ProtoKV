"""qwen_rvs_video.py

RVS / StreamingVQA inference with Qwen-VL and ProtoKV cache compression.
The external CUDA timing logger has been removed; the script now writes only
the prediction CSV requested by --output_csv.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import cv2

from transformers import (
    AutoProcessor,
    DynamicCache,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from kvcache_utils_proto import process_kv_cache, install_protokv_attention_bias_hook

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VIDEO_FORMATS = (".mp4", ".avi", ".mov", ".mkv")


class _NoOpTimer:
    """Small no-op context manager used after removing JSONL/CSV timing logs."""

    def __call__(self, name: str):
        return nullcontext()

    def reset(self) -> None:
        return None

    def snapshot(self) -> Dict[str, float]:
        return {}

    def add_cpu_seconds(self, name: str, seconds: float) -> None:
        return None

    def region_stats(self, name: str) -> Dict[str, float]:
        return {"total_ms": 0.0, "avg_ms": 0.0, "median_ms": 0.0, "count": 0.0}

    def is_enabled(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Data classes / helpers
# ---------------------------------------------------------------------------

@dataclass
class RVSSample:
    video_id: str
    video_path: str
    question: str
    answer: str
    answer_type: str
    start_time: float
    end_time: float
    fps_hint: Optional[float] = None


def _safe_mkdir(p: str) -> None:
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _cleanup_cuda_memory() -> None:
    """Release cached CUDA memory between samples and after OOM exceptions."""
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _linspace_indices(start: int, end_exclusive: int, n: int) -> List[int]:
    if end_exclusive <= start:
        end_exclusive = start + 1
    length = end_exclusive - start
    if n <= 0:
        return [start]
    if length <= n:
        return list(range(start, end_exclusive))
    idx = np.linspace(start, end_exclusive - 1, n)
    idx = np.round(idx).astype(int)
    out: List[int] = []
    last = None
    for i in idx.tolist():
        if last is None or i != last:
            out.append(i)
            last = i
    return out


@lru_cache(maxsize=16)
def _load_npy_mmap(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def _frame_to_pil_rgb(frame: np.ndarray) -> Image.Image:
    if frame.ndim != 3:
        raise ValueError(f"Expected 3D frame array, got shape={frame.shape}")
    if frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return Image.fromarray(frame)


def _resize_pil_to_max_pixels(img: Image.Image, max_pixels: int) -> Image.Image:
    """Resize one frame so W*H <= max_pixels, preserving aspect ratio.

    This is important because some Qwen processor versions ignore max_pixels
    embedded in the chat template when raw PIL frames are passed directly.
    Without this pre-resize, long videos can create >1M video tokens before
    ProtoKV compression begins.
    """
    max_pixels = int(max_pixels or 0)
    if max_pixels <= 0:
        return img.convert("RGB")
    w, h = img.size
    pixels = max(1, int(w) * int(h))
    if pixels <= max_pixels:
        return img.convert("RGB")
    scale = math.sqrt(float(max_pixels) / float(pixels))
    new_w = max(28, int(round(w * scale)))
    new_h = max(28, int(round(h * scale)))
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    return img.convert("RGB").resize((new_w, new_h), resampling)


# ---------------------------------------------------------------------------
# Frame loading helpers (unchanged)
# ---------------------------------------------------------------------------

def load_frames_from_npy(
    npy_path: str,
    start_time: float,
    end_time: float,
    max_frames: int,
    fps_hint: Optional[float] = None,
) -> List[Image.Image]:
    logger.info(f"Loading frames from .npy: {os.path.basename(npy_path)} "
                f"(time: {start_time:.2f}-{end_time:.2f}s)")
    arr = _load_npy_mmap(npy_path)
    if arr.ndim not in (4,):
        raise ValueError(f"Unsupported .npy shape: {arr.shape}")
    total_frames = int(arr.shape[0])
    fps = float(fps_hint) if fps_hint and fps_hint > 0 else 3.0
    start_idx = max(0, min(int(round(start_time * fps)), total_frames - 1))
    end_idx = max(start_idx + 1, min(int(round(end_time * fps)), total_frames))
    indices = _linspace_indices(start_idx, end_idx, max_frames)
    frames = [_frame_to_pil_rgb(np.asarray(arr[i])) for i in indices]
    logger.info(f"  Loaded {len(frames)} frames from .npy file")
    return frames


def load_frames_from_video_file(
    video_path: str,
    start_time: float,
    end_time: float,
    max_frames: int,
) -> List[Image.Image]:
    logger.info(f"Loading frames from video: {os.path.basename(video_path)} "
                f"(time: {start_time:.2f}-{end_time:.2f}s)")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_idx = int(start_time * fps)
    end_idx = int(end_time * fps)
    if total_frames > 0:
        start_idx = max(0, min(start_idx, total_frames - 1))
        end_idx = max(start_idx + 1, min(end_idx, total_frames))
    else:
        start_idx = max(0, start_idx)
        end_idx = max(start_idx + 1, end_idx)
    indices = _linspace_indices(start_idx, end_idx, max_frames)
    frames: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded for {video_path} in [{start_time}, {end_time}].")
    logger.info(f"  Loaded {len(frames)} frames from video file")
    return frames


def load_video_segment_frames(
    video_path: str,
    start_time: float,
    end_time: float,
    max_frames: int,
    fps_hint: Optional[float] = None,
) -> List[Image.Image]:
    ext = os.path.splitext(video_path)[1].lower()
    if ext == ".npy":
        return load_frames_from_npy(video_path, start_time, end_time, max_frames, fps_hint=fps_hint)
    if ext in VIDEO_FORMATS:
        return load_frames_from_video_file(video_path, start_time, end_time, max_frames)
    raise ValueError(f"Unsupported extension: {video_path}")


@lru_cache(maxsize=128)
def _get_video_duration_seconds(video_path: str, fps_hint: Optional[float] = None) -> Optional[float]:
    ext = os.path.splitext(video_path)[1].lower()
    if ext == ".npy":
        arr = _load_npy_mmap(video_path)
        total_frames = int(arr.shape[0]) if arr is not None else 0
        fps = float(fps_hint) if fps_hint and fps_hint > 0 else 3.0
        return total_frames / fps if total_frames > 0 and fps > 0 else None
    if ext in VIDEO_FORMATS:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fc = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = fc / fps if fc > 0 and fps > 0 else 0.0
        if duration <= 0:
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            duration = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        cap.release()
        return duration if duration > 0 else None
    return None


def load_rvs_samples(json_path: str, video_root: Optional[str] = None) -> List[RVSSample]:
    logger.info(f"Loading RVS samples from: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"  Found {len(data)} video items in JSON")
    samples: List[RVSSample] = []
    for item in data:
        video_id = str(item.get("video_id"))
        video_path = str(item.get("video_path"))
        if video_root and not os.path.isabs(video_path):
            video_path = os.path.join(video_root, video_path)
        clip_start = float(item.get("clip_start_time", 0.0) or 0.0)
        clip_end = item.get("clip_end_time", None)
        clip_end = float(clip_end) if clip_end is not None else None
        fps_hint: Optional[float] = None
        if video_path.lower().endswith(".npy"):
            try:
                conv0 = (item.get("conversations") or [])[0]
                dur = float(conv0.get("duration")) if conv0 and conv0.get("duration") else None
                if dur and clip_end is not None and clip_end > clip_start:
                    fps_hint = dur / (clip_end - clip_start)
            except Exception:
                fps_hint = None
            if fps_hint is None or fps_hint <= 0:
                fps_hint = 3.0
        for conv in item.get("conversations", []):
            q = str(conv.get("question", ""))
            a = str(conv.get("answer", ""))
            answer_type = str(conv.get("answer_type", ""))
            start_time = float(conv.get("start_time", 0.0) or 0.0)
            end_time = float(conv.get("end_time", 0.0) or 0.0)
            if "clip_start_time" in item:
                start_time -= clip_start
                end_time -= clip_start
            start_time = max(0.0, start_time)
            end_time = max(start_time + 1e-3, end_time)
            samples.append(RVSSample(
                video_id=video_id, video_path=video_path,
                question=q, answer=a, answer_type=answer_type,
                start_time=start_time, end_time=end_time, fps_hint=fps_hint,
            ))
    logger.info(f"  Parsed {len(samples)} total QA samples")
    return samples


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class RVSVideoEval:
    """Inference helper for RVS open-ended QA."""

    MAX_GEN_TOKENS = 300
    DEFAULT_EOS_TOKEN_ID = 151645
    DEFAULT_MAX_PIXELS = 16 * 28 * 28

    def __init__(
        self,
        model_path: str,
        max_frames_num: int = 32,
        max_pixels: Optional[int] = None,
        block_size: int = -1,
        compress_frame_num: int = 0,
        compression_method: str = "uniform",
        tar_ratio: float = 0.5,
        query_ratio: float = 0.25,
        adaptive_pooling: bool = False,
        load_dumped: bool = False,
        cache_dir: str = "cache/qwen_rvs_video_inputs",
        per_frame: bool = False,
        prototrack_proto_frames: int = 24,
        prototrack_pq_subspaces: int = 8,
        prototrack_pq_codebook_size: int = 16,
        prototrack_pq_kmeans_iters: int = 4,
        prototrack_pq_sample_size: int = 4096,
        prototrack_pq_seed: int = 0,
        prototrack_pq_modes: int = 8,
        prototrack_pq_beam_size: int = 0,
        prototrack_pq_beam_eps: float = 1e-5,
        prototrack_lambda_sp: float = 0.1,
        prototrack_lambda_idle: float = 0.01,
        prototrack_idle_threshold: int = 120,
        prototrack_alpha: float = 0.05,
        prototrack_beta: float = 0.05,
        prototrack_eta: float = 0.05,
        prototrack_maintenance_gamma: float = 0.05,
        prototrack_merge_eps_k: float = 0.20,
        prototrack_merge_eps_v: float = 0.25,
        prototrack_min_mass: float = 1.0,
        attn_implementation: str = "eager",
        gpu_max_memory_gib: float = 18.0,
        cpu_max_memory_gib: float = 64.0,
        verbose: bool = False,
    ) -> None:
        logger.info("Initializing RVSVideoEval")
        self.model_path = model_path
        self.max_frames_num = max_frames_num
        self.max_pixels = int(max_pixels or self.DEFAULT_MAX_PIXELS)
        self.block_size = block_size
        self.compress_frame_num = compress_frame_num
        self.compression_method = compression_method
        self.tar_ratio = tar_ratio
        self.query_ratio = query_ratio
        self.adaptive_pooling = adaptive_pooling
        self.load_dumped = load_dumped
        self.cache_dir = cache_dir
        self.per_frame = per_frame
        self.prototrack_proto_frames = int(prototrack_proto_frames)
        self.prototrack_pq_subspaces = int(prototrack_pq_subspaces)
        self.prototrack_pq_codebook_size = int(prototrack_pq_codebook_size)
        self.prototrack_pq_kmeans_iters = int(prototrack_pq_kmeans_iters)
        self.prototrack_pq_sample_size = int(prototrack_pq_sample_size)
        self.prototrack_pq_seed = int(prototrack_pq_seed)
        self.prototrack_pq_modes = int(prototrack_pq_modes)
        self.prototrack_pq_beam_size = int(prototrack_pq_beam_size)
        self.prototrack_pq_beam_eps = float(prototrack_pq_beam_eps)
        self.prototrack_lambda_sp = float(prototrack_lambda_sp)
        self.prototrack_lambda_idle = float(prototrack_lambda_idle)
        self.prototrack_idle_threshold = int(prototrack_idle_threshold)
        self.prototrack_alpha = float(prototrack_alpha)
        self.prototrack_beta = float(prototrack_beta)
        self.prototrack_eta = float(prototrack_eta)
        self.prototrack_maintenance_gamma = float(prototrack_maintenance_gamma)
        self.prototrack_merge_eps_k = float(prototrack_merge_eps_k)
        self.prototrack_merge_eps_v = float(prototrack_merge_eps_v)
        self.prototrack_min_mass = float(prototrack_min_mass)
        self.attn_implementation = str(attn_implementation)
        self.gpu_max_memory_gib = float(gpu_max_memory_gib or 0.0)
        self.cpu_max_memory_gib = float(cpu_max_memory_gib or 0.0)
        self.verbose = verbose

        # No external timing logger is used; keep a no-op timer so existing
        # context-manager calls do not change model behavior.
        self._timer = _NoOpTimer()

        logger.info(f"  Model: {model_path}")
        logger.info(f"  Max frames: {max_frames_num}")
        logger.info(f"  Max pixels/frame: {self.max_pixels}")
        if self.gpu_max_memory_gib > 0:
            logger.info(f"  Model GPU max memory: {self.gpu_max_memory_gib:.1f} GiB (CPU offload enabled via device_map=auto)")
        if block_size > 0:
            logger.info(f"  Block processing: enabled (block_size={block_size}, compress_frame_num={compress_frame_num})")
            logger.info(f"  Compression method: {compression_method}")
            if compression_method == "prototrack-kv":
                logger.info(
                    f"  ProtoTrack residual PQ: subspaces={self.prototrack_pq_subspaces}, "
                    f"codebook_size={self.prototrack_pq_codebook_size}, "
                    f"kmeans_iters={self.prototrack_pq_kmeans_iters}, "
                    f"sample_size={self.prototrack_pq_sample_size}, "
                    f"S={self.prototrack_pq_modes}, B={self.prototrack_pq_beam_size or 4 * self.prototrack_pq_modes}"
                )
                logger.info(
                    f"  ProtoKV paper EMA/assign: alpha={self.prototrack_alpha}, beta={self.prototrack_beta}, "
                    f"eta={self.prototrack_eta}, lambda_sp={self.prototrack_lambda_sp}, "
                    f"lambda_idle={self.prototrack_lambda_idle}, T_idle={self.prototrack_idle_threshold}"
                )
        else:
            logger.info("  Block processing: disabled")
        if self.load_dumped:
            _safe_mkdir(self.cache_dir)
            logger.info(f"  Caching enabled: {cache_dir}")

        self.model = None
        self.processor = None
        self._initialize_model()

    def _print(self, msg: str) -> None:
        if self.verbose:
            print(f">>> {msg}", flush=True)

    def _initialize_model(self) -> None:
        self._print(f"Loading model: {self.model_path}")
        logger.info("Loading model and processor...")
        max_memory = None
        if self.gpu_max_memory_gib > 0:
            # Keep several GiB free for Qwen's visual encoder, temporary activations,
            # and the KV cache. This avoids the common 24GB-GPU failure where the
            # model itself occupies ~90% of VRAM before video encoding begins.
            max_memory = {0: f"{self.gpu_max_memory_gib:.0f}GiB"}
            if self.cpu_max_memory_gib > 0:
                max_memory["cpu"] = f"{self.cpu_max_memory_gib:.0f}GiB"

        load_kwargs = dict(
            # Explicit dtype keeps FlashAttention2 and model weights in bf16 instead of
            # accidentally allocating fp32 tensors in some Transformers versions.
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=self.attn_implementation,
        )
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory

        if "2.5-vl" in self.model_path.lower():
            logger.info("  Detected Qwen2.5-VL model")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path, **load_kwargs
            )
        else:
            logger.info("  Detected Qwen2-VL model")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path, **load_kwargs
            )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self._print("Model loaded.")
        if self.compression_method == "prototrack-kv":
            install_protokv_attention_bias_hook(self.model)
            logger.info(f"ProtoKV bias-aware attention hook installed; attn_implementation={self.attn_implementation}")
        logger.info("Model and processor loaded successfully")
        try:
            self.model.eval()
        except Exception:
            pass

    def reset_timing(self) -> None:
        self._timer.reset()

    def get_timing(self) -> Dict[str, float]:
        return self._timer.snapshot()

    # ------------------------------------------------------------------
    # Layout helpers (unchanged)
    # ------------------------------------------------------------------

    def _compute_layout(self, inputs: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
        input_ids = inputs["input_ids"]
        video_token_id = self.model.config.video_token_id
        frame_number = int(inputs["video_grid_thw"][0, 0].item())
        if frame_number <= 0:
            raise ValueError(f"Invalid frame_number={frame_number}")
        mask_indices = (input_ids[0, :] == video_token_id).nonzero(as_tuple=True)[0]
        if mask_indices.numel() == 0:
            raise ValueError("No video_token_id found in input_ids")
        system_size = input_ids[0, : mask_indices[0] - 1].shape[-1] + 1
        inst_size = input_ids[0, mask_indices[-1] + 2 :].shape[-1] + 1
        token_per_frame = int((input_ids == video_token_id).sum().item() // frame_number)
        vision_length = input_ids.shape[1] - system_size - inst_size
        return system_size, inst_size, token_per_frame, frame_number, vision_length

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill_video_only(
        self, inputs: Dict[str, Any]
    ) -> Tuple[DynamicCache, torch.Tensor]:
        inputs.pop("second_per_grid_ts", None)
        input_ids = inputs["input_ids"]
        system_size, inst_size, token_per_frame, total_frames, vision_length = self._compute_layout(inputs)

        position_ids_full, _ = self.model.model.get_rope_index(
            input_ids, video_grid_thw=inputs["video_grid_thw"]
        )
        try:
            self.model.config.height_width = (
                position_ids_full[1, :, system_size : system_size + token_per_frame].max()
                - position_ids_full[1, :, system_size : system_size + token_per_frame].min()
                + 1
            )
        except Exception:
            pass

        past_key_values = DynamicCache()
        # Use block-wise prefill whenever possible. In per-frame mode this is
        # especially important: even if total_frames <= block_size, a one-shot
        # visual-encoder call can OOM on 24GB GPUs.
        use_block = (
            self.block_size > 0
            and self.compress_frame_num > 0
            and (total_frames > self.block_size or (self.per_frame and total_frames > 1))
        )

        if use_block:
            logger.info(f"Prefill mode: block-wise (frames={total_frames}, block_size={self.block_size}, keep={self.compress_frame_num})")
            past_key_values, position_ids_full = self._block_wise_prefill(
                inputs=inputs,
                input_ids=input_ids,
                past_key_values=past_key_values,
                system_size=system_size,
                inst_size=inst_size,
                token_per_frame=token_per_frame,
                vision_length=vision_length,
                total_frames=total_frames,
            )
        else:
            logger.info(f"Prefill mode: one-shot (frames={total_frames})")
            system_input_ids = input_ids[:, :system_size]
            system_embeds = self.model.model.get_input_embeddings()(system_input_ids)
            pixel_values_videos = inputs["pixel_values_videos"].to(self.model.device, dtype=self.model.dtype)
            video_grid_thw = inputs["video_grid_thw"]

            # ── CUDA event: encode ───────────────────────────────────────
            with self._timer("encode"):
                with torch.inference_mode():
                    video_embeds = self.model.get_video_features(
                        pixel_values_videos, video_grid_thw=video_grid_thw
                    )
                    video_embeds = torch.cat(video_embeds, dim=0)

            video_embeds = video_embeds.to(system_embeds.device, system_embeds.dtype)
            inputs_embeds = torch.cat([system_embeds, video_embeds.unsqueeze(0)], dim=1)
            token_end = system_size + total_frames * token_per_frame
            position_ids = position_ids_full[:, :, :token_end]

            # ── CUDA event: prefill ──────────────────────────────────────
            with self._timer("prefill"):
                with torch.inference_mode():
                    outputs = self.model(
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=True,
                    )
            past_key_values = outputs[1]

        return past_key_values, position_ids_full

    # ------------------------------------------------------------------
    # Decode from prefilled cache
    # ------------------------------------------------------------------

    def answer_from_prefill(
        self,
        inputs: Dict[str, Any],
        past_key_values: DynamicCache,
        position_ids_full: torch.Tensor,
    ) -> Tuple[str, float, float]:
        """Reveal question and decode.

        Returns
        -------
        pred_answer : str
        ttft_ms     : float  (time-to-first-token, milliseconds)
        e2e_ms      : float  (end-to-end decode, milliseconds)
        """
        input_ids = inputs["input_ids"]
        video_token_id = self.model.config.video_token_id
        mask_indices = (input_ids[0, :] == video_token_id).nonzero(as_tuple=True)[0]
        inst_start = mask_indices[-1] + 1
        post_input_ids = input_ids[:, inst_start:]
        prompt_len = post_input_ids.shape[-1]

        pos_1d = torch.arange(prompt_len, device=post_input_ids.device).expand(post_input_ids.shape[0], -1)
        base = position_ids_full[0, 0, -1] + 1
        position_ids = (pos_1d + base).unsqueeze(0).expand(3, -1, -1)

        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", self.DEFAULT_EOS_TOKEN_ID)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_start = time.perf_counter()

        input_ids_save = post_input_ids
        input_ids_current = post_input_ids
        ttft_ms = float("nan")

        for step in range(self.MAX_GEN_TOKENS):
            with torch.inference_mode():
                outputs = self.model(
                    input_ids_current,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    use_cache=True,
                )
            past_key_values = outputs[1]
            next_token = outputs[0][:, -1, :].argmax(dim=-1).unsqueeze(1)

            if step == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ttft_ms = (time.perf_counter() - decode_start) * 1000.0

            input_ids_save = torch.cat([input_ids_save, next_token], dim=-1)
            input_ids_current = next_token
            position_ids = position_ids[:, :, -1:] + 1

            if next_token[0, -1].item() == eos_token_id:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        e2e_ms = (time.perf_counter() - decode_start) * 1000.0

        output_ids = input_ids_save[:, prompt_len:]
        pred = self.processor.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return pred, ttft_ms, e2e_ms

    # ------------------------------------------------------------------
    # Caching helpers (unchanged logic, removed old _add_timing calls)
    # ------------------------------------------------------------------

    def _dump_path(self, video_path: str, start_time: float, end_time: float) -> str:
        key = f"{video_path}|{start_time:.3f}|{end_time:.3f}|{self.max_frames_num}|{self.max_pixels}"
        return os.path.join(self.cache_dir, _md5(key) + ".pt")

    def _sanitize_processor_outputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only required video tensors; keep large video pixels on CPU."""
        clean: Dict[str, Any] = {}
        for k, v in inputs.items():
            if k in ("input_ids", "attention_mask"):
                # Rebuilt below from chat-template tokens and visual-token count.
                continue
            if torch.is_tensor(v):
                if k == "pixel_values_videos":
                    clean[k] = v.detach().cpu()
                else:
                    clean[k] = v.to(self.model.device)
            else:
                clean[k] = v
        if "pixel_values_videos" not in clean or "video_grid_thw" not in clean:
            raise KeyError(f"Processor output missing video tensors. Got keys={list(inputs.keys())}")
        return clean

    def _load_or_process_video(
        self,
        text: str,
        frames: List[Image.Image],
        dump_path: str,
        is_first_sample: bool,
    ) -> Dict[str, Any]:
        if self.load_dumped and os.path.exists(dump_path):
            logger.debug(f"  Loading cached processor inputs: {os.path.basename(dump_path)}")
            inputs = torch.load(dump_path, map_location="cpu")
            return self._sanitize_processor_outputs(inputs)

        logger.info(f"  Processing {len(frames)} frames with processor...")
        frames = [_resize_pil_to_max_pixels(f, self.max_pixels) for f in frames]
        cpu_t0 = time.perf_counter()
        try:
            inputs = self.processor(
                text=[text], images=None, videos=[frames],
                padding=True, return_tensors="pt",
                max_pixels=self.max_pixels,
            )
        except TypeError:
            inputs = self.processor(
                text=[text], images=None, videos=[frames],
                padding=True, return_tensors="pt",
            )
        cpu_elapsed_s = time.perf_counter() - cpu_t0
        logger.info(f"  Processor completed in {cpu_elapsed_s:.3f}s")
        if is_first_sample:
            self._print(f"Processed segment in {cpu_elapsed_s:.3f}s")

        inputs = self._sanitize_processor_outputs(inputs)
        if self.load_dumped:
            torch.save({k: (v.cpu() if torch.is_tensor(v) else v) for k, v in inputs.items()}, dump_path)

        return inputs

    def prepare_video_input(
        self,
        video_path: str,
        question_text: str,
        start_time: float,
        end_time: float,
        fps_hint: Optional[float] = None,
        is_first_sample: bool = False,
    ) -> Dict[str, Any]:
        logger.info(f"Preparing video input for: {os.path.basename(video_path)}")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "<VIDEO>",
                     "max_frames": self.max_frames_num,
                     "max_pixels": self.max_pixels},
                    {"type": "text", "text": question_text},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        frames = load_video_segment_frames(video_path, start_time, end_time, self.max_frames_num, fps_hint=fps_hint)
        dump_path = self._dump_path(video_path, start_time, end_time)
        inputs = self._load_or_process_video(text=text, frames=frames, dump_path=dump_path, is_first_sample=is_first_sample)
        input_ids = self.processor.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        video_length = inputs["pixel_values_videos"].shape[0] // 4
        frame_count = int(inputs["video_grid_thw"][0, 0].item())
        logger.info(f"  Visual tokens after resize/processor: {video_length} ({video_length / max(1, frame_count):.1f} per frame)")
        video_pad_tokens = torch.full(
            (video_length,), self.model.config.video_token_id,
            dtype=torch.long, device=self.model.device,
        ).unsqueeze(0)
        vision_start_idx = (input_ids == self.model.config.vision_start_token_id).nonzero(as_tuple=True)[-1]
        vision_end_idx = (input_ids == self.model.config.vision_end_token_id).nonzero(as_tuple=True)[-1]
        inputs["input_ids"] = torch.cat(
            [input_ids[:, : vision_start_idx + 1], video_pad_tokens, input_ids[:, vision_end_idx:]], dim=-1
        )
        inputs["attention_mask"] = torch.ones(inputs["input_ids"].shape, device=self.model.device)
        return inputs

    # ------------------------------------------------------------------
    # Standard generate (no block processing)
    # ------------------------------------------------------------------

    def generate(self, inputs: Dict[str, Any]) -> str:
        """Standard (non-block) generation with explicit encode / prefill / decode phases."""
        logger.info("Running standard generation...")
        inputs.pop("second_per_grid_ts", None)

        input_ids    = inputs["input_ids"]
        pixel_values = inputs["pixel_values_videos"].to(self.model.device, dtype=self.model.dtype)
        video_grid   = inputs["video_grid_thw"]

        # ── 1. Encode: visual tokens → embeddings ───────────────────────────
        with self._timer("encode"):
            with torch.inference_mode():
                video_embeds = self.model.get_video_features(pixel_values, video_grid_thw=video_grid)
                video_embeds = torch.cat(video_embeds, dim=0)

        # Build full inputs_embeds (system + video tokens only; question appended below)
        video_token_id    = self.model.config.video_token_id
        vision_start_id   = self.model.config.vision_start_token_id
        vision_end_id     = self.model.config.vision_end_token_id

        mask_start = (input_ids == vision_start_id).nonzero(as_tuple=True)[-1]
        mask_end   = (input_ids == vision_end_id).nonzero(as_tuple=True)[-1]

        embed_fn = self.model.model.get_input_embeddings()
        pre_embeds  = embed_fn(input_ids[:, : mask_start + 1])          # up to <vision_start>
        post_embeds = embed_fn(input_ids[:, mask_end:])                  # <vision_end> onwards (question + asst prompt)
        inputs_embeds = torch.cat([pre_embeds, video_embeds.unsqueeze(0), post_embeds], dim=1)

        # ── 2. Prefill: one forward pass, cache the KV ──────────────────────
        position_ids_full, _ = self.model.model.get_rope_index(
            input_ids, video_grid_thw=video_grid
        )

        with self._timer("prefill"):
            with torch.inference_mode():
                prefill_out = self.model(
                    inputs_embeds=inputs_embeds,
                    position_ids=position_ids_full,
                    use_cache=True,
                )
        past_key_values = prefill_out[1]

        # ── 3. Decode: autoregressive loop ───────────────────────────────────
        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", self.DEFAULT_EOS_TOKEN_ID)
        next_token   = prefill_out[0][:, -1, :].argmax(dim=-1, keepdim=True)  # first generated token
        generated    = [next_token]
        pos          = (position_ids_full[:, :, -1:] + 1)                     # continue positions

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_t0 = time.perf_counter()

        with torch.inference_mode():
            for _ in range(self.MAX_GEN_TOKENS - 1):
                out = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    position_ids=pos,
                    use_cache=True,
                )
                past_key_values = out[1]
                next_token = out[0][:, -1, :].argmax(dim=-1, keepdim=True)
                generated.append(next_token)
                pos = pos[:, :, -1:] + 1
                if next_token[0, -1].item() == eos_token_id:
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_ms = (time.perf_counter() - dec_t0) * 1000.0
        logger.info(f"  Decode: {len(generated)} tokens in {dec_ms:.1f} ms")

        output_ids = torch.cat(generated, dim=-1)
        return self.processor.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    # ------------------------------------------------------------------
    # Block-wise visual encoding
    # ------------------------------------------------------------------

    def _visual_encoding_block(
        self,
        pixel_values_videos: torch.Tensor,
        video_grid_thw: torch.Tensor,
        frame_start: int,
        frame_end: int,
        total_frames: int,
    ) -> torch.Tensor:
        logger.debug(f"    Encoding visual block: frames [{frame_start}, {frame_end})")
        total_patches = pixel_values_videos.shape[0]
        patches_per_frame = total_patches // total_frames
        patch_start = frame_start * patches_per_frame
        patch_end = frame_end * patches_per_frame
        block_pixel_values = pixel_values_videos[patch_start:patch_end]
        block_frame_num = frame_end - frame_start
        block_video_grid_thw = video_grid_thw.clone().to(self.model.device)
        block_video_grid_thw[0, 0] = block_frame_num
        block_pixel_values = block_pixel_values.to(self.model.device, dtype=self.model.dtype, non_blocking=True)

        with self._timer("encode"):
            with torch.inference_mode():
                video_embeds = self.model.get_video_features(
                    block_pixel_values, video_grid_thw=block_video_grid_thw
                )
                video_embeds = torch.cat(video_embeds, dim=0)
        return video_embeds

    # ------------------------------------------------------------------
    # Block-wise prefill
    # ------------------------------------------------------------------

    def _block_wise_prefill(
        self,
        inputs: Dict[str, Any],
        input_ids: torch.Tensor,
        past_key_values: DynamicCache,
        system_size: int,
        inst_size: int,
        token_per_frame: int,
        vision_length: int,
        total_frames: int,
    ) -> Tuple[DynamicCache, torch.Tensor]:
        mode = "frame-wise" if self.per_frame else "block-wise"
        logger.info(f"  {mode.capitalize()} prefill: {total_frames} frames, block_size={self.block_size}")
        cur_frame = 0
        system_input_ids = input_ids[:, :system_size]
        system_embeds = self.model.model.get_input_embeddings()(system_input_ids)
        pixel_values_videos = inputs["pixel_values_videos"]
        video_grid_thw = inputs["video_grid_thw"]
        position_ids_full, _ = self.model.model.get_rope_index(input_ids, video_grid_thw=video_grid_thw)
        self.model.config.height_width = (
            position_ids_full[1, :, system_size : system_size + token_per_frame].max()
            - position_ids_full[1, :, system_size : system_size + token_per_frame].min()
            + 1
        )

        while cur_frame < total_frames:
            frame_start = cur_frame
            if self.per_frame:
                frames_in_block = 1
            elif frame_start == 0:
                frames_in_block = min(self.block_size, total_frames)
            else:
                frames_in_block = min(
                    self.block_size - self.compress_frame_num,
                    total_frames - frame_start,
                )
            frame_end = frame_start + frames_in_block
            is_first_block = frame_start == 0
            is_last_block = frame_end >= total_frames
            logger.info(f"  Processing block: frames [{frame_start}, {frame_end}), first={is_first_block}, last={is_last_block}")

            block_video_embeds = self._visual_encoding_block(
                pixel_values_videos, video_grid_thw, frame_start, frame_end, total_frames
            )
            block_video_embeds = block_video_embeds.to(system_embeds.device, system_embeds.dtype)

            if is_first_block:
                block_inputs_embeds = torch.cat([system_embeds, block_video_embeds.unsqueeze(0)], dim=1)
                token_start = 0
                token_end = system_size + frames_in_block * token_per_frame
            else:
                block_inputs_embeds = block_video_embeds.unsqueeze(0)
                token_start = system_size + frame_start * token_per_frame
                token_end = system_size + frame_end * token_per_frame

            position_ids = position_ids_full[:, :, token_start:token_end]

            # ── CUDA event: prefill ──────────────────────────────────────
            with self._timer("prefill"):
                with torch.inference_mode():
                    outputs = self.model(
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        inputs_embeds=block_inputs_embeds,
                        use_cache=True,
                    )
            past_key_values = outputs[1]

            should_compress = not is_last_block
            if self.per_frame:
                # In frame-wise mode, start compression after warmup reaches block_size.
                should_compress = should_compress and (frame_end >= self.block_size)

            if should_compress:
                res_frame_num = total_frames - frame_end
                compress_frame_num = (
                    self.compress_frame_num
                    if (self.compress_frame_num + res_frame_num) >= self.block_size
                    else self.block_size - res_frame_num
                )
                logger.debug(f"    Compressing KV cache: method={self.compression_method}, compress_frame_num={compress_frame_num}")

                # ── CUDA event: compress ─────────────────────────────────
                with self._timer("compress"):
                    with torch.no_grad():
                        past_key_values, _ = process_kv_cache(
                            past_key_values=past_key_values,
                            model=self.model,
                            system_size=system_size,
                            inst_size=inst_size,
                            token_per_frame=token_per_frame,
                            compress_frame_num=compress_frame_num,
                            method=self.compression_method,
                            tar_ratio=self.tar_ratio,
                            query_ratio=self.query_ratio,
                            adaptive_pooling=self.adaptive_pooling,
                            is_first_block=is_first_block,
                            is_last_block=is_last_block,
                            per_frame=self.per_frame,
                            prototrack_proto_frames=self.prototrack_proto_frames,
                            prototrack_frame_end=frame_end,
                            prototrack_pq_subspaces=self.prototrack_pq_subspaces,
                            prototrack_pq_codebook_size=self.prototrack_pq_codebook_size,
                            prototrack_pq_kmeans_iters=self.prototrack_pq_kmeans_iters,
                            prototrack_pq_sample_size=self.prototrack_pq_sample_size,
                            prototrack_pq_seed=self.prototrack_pq_seed,
                            prototrack_pq_modes=self.prototrack_pq_modes,
                            prototrack_pq_beam_size=self.prototrack_pq_beam_size,
                            prototrack_pq_beam_eps=self.prototrack_pq_beam_eps,
                            prototrack_lambda_sp=self.prototrack_lambda_sp,
                            prototrack_lambda_idle=self.prototrack_lambda_idle,
                            prototrack_idle_threshold=self.prototrack_idle_threshold,
                            prototrack_alpha=self.prototrack_alpha,
                            prototrack_beta=self.prototrack_beta,
                            prototrack_eta=self.prototrack_eta,
                            prototrack_maintenance_gamma=self.prototrack_maintenance_gamma,
                            prototrack_merge_eps_k=self.prototrack_merge_eps_k,
                            prototrack_merge_eps_v=self.prototrack_merge_eps_v,
                            prototrack_min_mass=self.prototrack_min_mass,
                        )

            cur_frame = frame_end

        return past_key_values, position_ids_full

    # ------------------------------------------------------------------
    # Generation stage (block path)
    # ------------------------------------------------------------------

    def _generation_stage(
        self,
        inputs: Dict[str, Any],
        past_key_values: DynamicCache,
        position_ids_full: torch.Tensor,
    ) -> str:
        logger.info("  Generation stage: decoding response...")
        input_ids = inputs["input_ids"]
        video_token_id = self.model.config.video_token_id
        mask_indices = (input_ids[0, :] == video_token_id).nonzero(as_tuple=True)[0]
        inst_start = mask_indices[-1] + 1
        post_input_ids = input_ids[:, inst_start:]
        input_len = post_input_ids.shape[-1]
        input_ids_save = post_input_ids
        input_ids_current = post_input_ids
        position_ids = (
            torch.arange(input_len, device=input_ids_current.device)
            .expand(input_ids_current.shape[0], -1)
        )
        position_ids = (position_ids + position_ids_full[0, 0, -1] + 1).unsqueeze(0).expand(3, -1, -1)
        eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", self.DEFAULT_EOS_TOKEN_ID)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_t0 = time.perf_counter()

        token_count = 0
        for _ in range(self.MAX_GEN_TOKENS):
            with torch.no_grad():
                outputs = self.model(
                    input_ids_current,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
            past_key_values = outputs[1]
            next_token = outputs[0][:, -1, :].argmax(dim=-1).unsqueeze(1)
            input_ids_save = torch.cat([input_ids_save, next_token], dim=-1)
            input_ids_current = next_token
            position_ids = position_ids[:, :, -1:] + 1
            token_count += 1
            if next_token[0, -1] == eos_token_id:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_ms = (time.perf_counter() - dec_t0) * 1000.0
        tok_s = token_count / max(dec_ms / 1000.0, 1e-9)
        logger.info(f"  Generated {token_count} tokens in {dec_ms:.1f} ms ({tok_s:.1f} tok/s)")

        output_ids = input_ids_save[:, input_len:]
        return self.processor.tokenizer.decode(
            output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def block_process(self, inputs: Dict[str, Any]) -> str:
        logger.info("Running block-wise processing...")
        input_ids = inputs["input_ids"]
        video_token_id = self.model.config.video_token_id
        frame_number = int(inputs["video_grid_thw"][0, 0].item())
        mask_indices = (input_ids[0, :] == video_token_id).nonzero(as_tuple=True)[0]
        system_size = input_ids[0, : mask_indices[0] - 1].shape[-1] + 1
        inst_size = input_ids[0, mask_indices[-1] + 2 :].shape[-1] + 1
        token_per_frame = int((input_ids == video_token_id).sum() // frame_number)
        vision_length = input_ids.shape[1] - system_size - inst_size
        logger.info(f"  Total frames: {frame_number}, token_per_frame: {token_per_frame}")
        logger.info(f"  System tokens: {system_size}, Instruction tokens: {inst_size}, Vision tokens: {vision_length}")
        past_key_values = DynamicCache()
        past_key_values, position_ids_full = self._block_wise_prefill(
            inputs, input_ids, past_key_values,
            system_size, inst_size, token_per_frame, vision_length, frame_number,
        )
        return self._generation_stage(inputs, past_key_values, position_ids_full)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 80)
    logger.info("Starting RVS (StreamingVQA) inference with Qwen2-VL")
    logger.info("=" * 80)

    parser = argparse.ArgumentParser(description="RVS (StreamingVQA) inference with Qwen2-VL")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--video_root", type=str, default=None)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--experiment", type=str, default="standard",
                        choices=["standard", "query_delay"])
    parser.add_argument("--deltas", type=str, default="0")
    parser.add_argument("--clip_mode", type=str, default="window",
                        choices=["window", "prefix"])
    parser.add_argument("--max_frames_num", type=int, default=768)
    parser.add_argument("--max_pixels", type=int, default=16 * 28 * 28,
                        help="Maximum pixels per frame before Qwen processing. Lower this if the visual encoder still OOMs. Safe 24GB default: 12544 (=16*28*28).")
    parser.add_argument("--gpu_max_memory_gib", type=float, default=18.0,
                        help="Maximum GPU memory used to load model weights with device_map=auto. Lower leaves more VRAM for visual encoding/KV cache; 18 is safer for 24GB GPUs.")
    parser.add_argument("--cpu_max_memory_gib", type=float, default=64.0,
                        help="CPU memory budget for model offload when --gpu_max_memory_gib is set.")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--compress_frame_num", type=int, default=0)
    parser.add_argument("--compression_method", type=str, default="uniform")
    parser.add_argument("--tar_ratio", type=float, default=0.5)
    parser.add_argument("--query_ratio", type=float, default=0.25)
    parser.add_argument("--adaptive_pooling", action="store_true")
    parser.add_argument(
        "--per_frame",
        action="store_true",
        help="Use frame-wise block prefill (1 frame per prefill step) and per-frame compression after warmup.",
    )
    parser.add_argument("--prototrack_proto_frames", type=int, default=18)
    parser.add_argument("--prototrack_pq_subspaces", type=int, default=8,
                        help="Number of PQ subquantizers for ProtoTrack residual histograms. Set 0 to disable residual PQ.")
    parser.add_argument("--prototrack_pq_codebook_size", type=int, default=16,
                        help="Number of codewords per PQ subquantizer for ProtoTrack residual histograms.")
    parser.add_argument("--prototrack_pq_kmeans_iters", type=int, default=4,
                        help="Mini-batch k-means iterations for residual PQ codebook initialization.")
    parser.add_argument("--prototrack_pq_sample_size", type=int, default=4096,
                        help="Maximum residual vectors sampled to initialize each residual PQ codebook.")
    parser.add_argument("--prototrack_pq_seed", type=int, default=0,
                        help="Random seed for residual PQ initialization.")
    parser.add_argument("--prototrack_pq_modes", type=int, default=8,
                        help="S: number of top-S residual modes / pseudo-token groups per prototype. Paper default: 8.")
    parser.add_argument("--prototrack_pq_beam_size", type=int, default=0,
                        help="B: beam size for DecodeTopSResidualModes. If 0, use B=4S. Paper default with S=8: 32.")
    parser.add_argument("--prototrack_pq_beam_eps", type=float, default=1e-5,
                        help="Epsilon smoothing for PQ histogram decoding.")
    parser.add_argument("--prototrack_lambda_sp", type=float, default=0.1)
    parser.add_argument("--prototrack_lambda_idle", type=float, default=0.01)
    parser.add_argument("--prototrack_idle_threshold", type=int, default=120)
    parser.add_argument("--prototrack_alpha", type=float, default=0.05)
    parser.add_argument("--prototrack_beta", type=float, default=0.05)
    parser.add_argument("--prototrack_eta", type=float, default=0.05)
    parser.add_argument("--prototrack_maintenance_gamma", type=float, default=0.05)
    parser.add_argument("--prototrack_merge_eps_k", type=float, default=0.20)
    parser.add_argument("--prototrack_merge_eps_v", type=float, default=0.25)
    parser.add_argument("--prototrack_min_mass", type=float, default=1.0)
    parser.add_argument("--attn_implementation", type=str, default="eager",
                        choices=["eager", "sdpa", "flash_attention_2"],
                        help="Use eager for exact ProtoKV log-mass bias. FlashAttention2 may ignore arbitrary positive additive bias.")
    parser.add_argument("--load_dumped", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="cache/qwen_rvs_video_inputs")


    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger.info("\nConfiguration:")
    logger.info(f"  Data path: {args.data_path}")
    logger.info(f"  Output CSV: {args.output_csv}")
    logger.info(f"  Experiment: {args.experiment}")
    logger.info(f"  Max frames: {args.max_frames_num}")
    logger.info(f"  Max pixels/frame: {args.max_pixels}")
    logger.info(f"  GPU max memory for model: {args.gpu_max_memory_gib} GiB")
    evaluator = RVSVideoEval(
        model_path=args.model_path,
        max_frames_num=args.max_frames_num,
        max_pixels=args.max_pixels,
        block_size=args.block_size,
        compress_frame_num=args.compress_frame_num,
        compression_method=args.compression_method,
        tar_ratio=args.tar_ratio,
        query_ratio=args.query_ratio,
        adaptive_pooling=args.adaptive_pooling,
        load_dumped=args.load_dumped,
        cache_dir=args.cache_dir,
        per_frame=args.per_frame,
        prototrack_proto_frames=args.prototrack_proto_frames,
        prototrack_pq_subspaces=args.prototrack_pq_subspaces,
        prototrack_pq_codebook_size=args.prototrack_pq_codebook_size,
        prototrack_pq_kmeans_iters=args.prototrack_pq_kmeans_iters,
        prototrack_pq_sample_size=args.prototrack_pq_sample_size,
        prototrack_pq_seed=args.prototrack_pq_seed,
        prototrack_pq_modes=args.prototrack_pq_modes,
        prototrack_pq_beam_size=args.prototrack_pq_beam_size,
        prototrack_pq_beam_eps=args.prototrack_pq_beam_eps,
        prototrack_lambda_sp=args.prototrack_lambda_sp,
        prototrack_lambda_idle=args.prototrack_lambda_idle,
        prototrack_idle_threshold=args.prototrack_idle_threshold,
        prototrack_alpha=args.prototrack_alpha,
        prototrack_beta=args.prototrack_beta,
        prototrack_eta=args.prototrack_eta,
        prototrack_maintenance_gamma=args.prototrack_maintenance_gamma,
        prototrack_merge_eps_k=args.prototrack_merge_eps_k,
        prototrack_merge_eps_v=args.prototrack_merge_eps_v,
        prototrack_min_mass=args.prototrack_min_mass,
        attn_implementation=args.attn_implementation,
        gpu_max_memory_gib=args.gpu_max_memory_gib,
        cpu_max_memory_gib=args.cpu_max_memory_gib,
        verbose=args.verbose,
    )

    samples = load_rvs_samples(args.data_path, video_root=args.video_root)
    logger.info(f"\nStarting inference on {len(samples)} samples...")
    _safe_mkdir(os.path.dirname(args.output_csv) or ".")

    try:
        delta_minutes_list = [float(x.strip()) for x in args.deltas.split(",") if x.strip()]
    except Exception as e:
        raise ValueError(f"Failed to parse --deltas='{args.deltas}'.") from e

    fieldnames_standard = ["video_id", "question", "answer", "pred_answer"]
    fieldnames_delay = ["video_id", "question", "answer", "pred_answer",
                        "answer_type", "delta", "ttft_ms", "e2e_ms"]
    fieldnames = fieldnames_delay if args.experiment == "query_delay" else fieldnames_standard

    with open(args.output_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()

        for i, s in enumerate(tqdm(samples, desc="RVS inference", dynamic_ncols=True)):
            is_first = i == 0
            logger.info(f"\n[Sample {i+1}/{len(samples)}] video_id={s.video_id}")

            # ----------------------------------------------------------
            # Standard experiment
            # ----------------------------------------------------------
            if args.experiment != "query_delay":
                evaluator.reset_timing()
                wall_start = time.perf_counter()
                frame_count = 0
                try:
                    inputs = evaluator.prepare_video_input(
                        video_path=s.video_path,
                        question_text=s.question,
                        start_time=0.0 if args.clip_mode == "prefix" else s.start_time,
                        end_time=s.end_time,
                        fps_hint=s.fps_hint,
                        is_first_sample=is_first,
                    )
                    frame_count = int(inputs["video_grid_thw"][0, 0].item())
                    use_block = (
                        evaluator.block_size > 0
                        and evaluator.compress_frame_num > 0
                        and (frame_count > evaluator.block_size or (evaluator.per_frame and frame_count > 1))
                    )
                    response = evaluator.block_process(inputs) if use_block else evaluator.generate(inputs)
                    writer.writerow({
                        "video_id": s.video_id, "question": s.question,
                        "answer": s.answer, "pred_answer": (response or "").strip(),
                    })
                except Exception as e:
                    logger.exception(f"Error for video_id={s.video_id}: {e}")
                    writer.writerow({
                        "video_id": s.video_id, "question": s.question,
                        "answer": s.answer, "pred_answer": "",
                    })
                finally:
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                    _cleanup_cuda_memory()
                    total_wall_ms = (time.perf_counter() - wall_start) * 1000.0
                    logger.info(f"  Finished sample in {total_wall_ms:.1f} ms")
                continue

            # ----------------------------------------------------------
            # Query-delay experiment
            # ----------------------------------------------------------
            t0 = float(s.end_time)
            video_duration = _get_video_duration_seconds(s.video_path, s.fps_hint)

            for delta_min in delta_minutes_list:
                evaluator.reset_timing()
                wall_start = time.perf_counter()
                frame_count = 0
                ttft_ms = float("nan")
                e2e_ms = float("nan")
                try:
                    t = t0 + float(delta_min) * 60.0
                    if video_duration is not None and t > video_duration:
                        logger.info(f"  [Delta={delta_min}m] t={t:.1f}s > duration {video_duration:.1f}s; clamping.")
                        t = video_duration

                    inputs = evaluator.prepare_video_input(
                        video_path=s.video_path,
                        question_text=s.question,
                        start_time=0.0,
                        end_time=t,
                        fps_hint=s.fps_hint,
                        is_first_sample=is_first,
                    )
                    frame_count = int(inputs["video_grid_thw"][0, 0].item())
                    past_kv, pos_full = evaluator.prefill_video_only(inputs)
                    pred, ttft_ms, e2e_ms = evaluator.answer_from_prefill(inputs, past_kv, pos_full)

                    writer.writerow({
                        "video_id": s.video_id, "question": s.question,
                        "answer": s.answer, "pred_answer": (pred or "").strip(),
                        "answer_type": s.answer_type, "delta": delta_min,
                        "ttft_ms": f"{ttft_ms:.3f}", "e2e_ms": f"{e2e_ms:.3f}",
                    })
                except Exception as e:
                    logger.exception(f"Error for video_id={s.video_id} delta={delta_min}: {e}")
                    writer.writerow({
                        "video_id": s.video_id, "question": s.question,
                        "answer": s.answer, "pred_answer": "",
                        "answer_type": s.answer_type, "delta": delta_min,
                        "ttft_ms": "", "e2e_ms": "",
                    })
                finally:
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception:
                            pass
                    _cleanup_cuda_memory()
                    total_wall_ms = (time.perf_counter() - wall_start) * 1000.0
                    logger.info(
                        f"  [Delta={delta_min}m] Finished sample in {total_wall_ms:.1f} ms "
                        f"(ttft={ttft_ms:.1f} ms, e2e={e2e_ms:.1f} ms)"
                    )

    logger.info("\n" + "=" * 80)
    logger.info(f"Inference complete! Processed {len(samples)} samples")
    logger.info(f"Predictions → {args.output_csv}")
    logger.info("=" * 80)
    print(f"Saved predictions to: {args.output_csv}")


if __name__ == "__main__":
    main()
