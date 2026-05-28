from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any
import inspect
import math
import time

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class _ProtoTrackState:
    max_proto_frames: int
    counts: List[torch.Tensor]              # [H, P]
    active: List[torch.Tensor]              # [H, P] bool
    taus: List[torch.Tensor]                # [H, P]
    mus: List[Optional[torch.Tensor]]       # [H, P, TPF, 2]
    sigmas: List[Optional[torch.Tensor]]    # [H, P, TPF, 2, 2]
    initialized: List[bool]
    proto_frames_cur: List[int]
    total_frames_seen: int = 0
    pq_codebooks_k: Optional[List[Optional[torch.Tensor]]] = None  # [H,G,C,ds]
    pq_codebooks_v: Optional[List[Optional[torch.Tensor]]] = None
    pq_hist_k: Optional[List[Optional[torch.Tensor]]] = None       # [H,P,TPF,G,C]
    pq_hist_v: Optional[List[Optional[torch.Tensor]]] = None
    pq_res_counts: Optional[List[Optional[torch.Tensor]]] = None   # [H,P,TPF]
    # Persistent prototype centers required for exact Algorithm-1/2 state.
    # They are not recovered from decoded pseudo-token cache outputs.
    proto_keys: Optional[List[Optional[torch.Tensor]]] = None       # [H,P,TPF,D]
    proto_values: Optional[List[Optional[torch.Tensor]]] = None
    pq_hist_token_per_frame: int = 0
    pq_last_subspaces: int = 0
    pq_last_codebook_size: int = 0
    bias_by_layer: Optional[List[Optional[torch.Tensor]]] = None   # [1,H,L]


class _NullCtx:
    def __init__(self, timing: Optional[Dict[str, float]] = None) -> None:
        self._timing = timing
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        if self._timing is not None:
            self._timing["proto_update_s"] = self._timing.get("proto_update_s", 0.0) + (time.perf_counter() - self._t0)


def _new_state(num_layers: int, num_heads: int, max_proto_frames: int, device) -> _ProtoTrackState:
    counts = [torch.zeros((num_heads, max_proto_frames), device=device, dtype=torch.float32) for _ in range(num_layers)]
    active = [torch.zeros((num_heads, max_proto_frames), device=device, dtype=torch.bool) for _ in range(num_layers)]
    taus = [torch.zeros((num_heads, max_proto_frames), device=device, dtype=torch.long) for _ in range(num_layers)]
    return _ProtoTrackState(
        max_proto_frames=int(max_proto_frames),
        counts=counts,
        active=active,
        taus=taus,
        mus=[None for _ in range(num_layers)],
        sigmas=[None for _ in range(num_layers)],
        initialized=[False for _ in range(num_layers)],
        proto_frames_cur=[0 for _ in range(num_layers)],
        pq_codebooks_k=[None for _ in range(num_layers)],
        pq_codebooks_v=[None for _ in range(num_layers)],
        pq_hist_k=[None for _ in range(num_layers)],
        pq_hist_v=[None for _ in range(num_layers)],
        pq_res_counts=[None for _ in range(num_layers)],
        proto_keys=[None for _ in range(num_layers)],
        proto_values=[None for _ in range(num_layers)],
        bias_by_layer=[None for _ in range(num_layers)],
    )




def clear_protokv_attention_bias(model) -> None:
    """Clear any stale ProtoKV attention-bias state from a previous sample.

    The attention hook is persistent once installed, but the bias itself is
    sample-specific. Clearing prevents a previous compressed video from
    influencing a later uncompressed / baseline run.
    """
    if hasattr(model, "_protokv_bias_by_layer"):
        model._protokv_bias_by_layer = None
    state = getattr(model, "_prototrack_kv_state", None)
    if isinstance(state, _ProtoTrackState) and state.bias_by_layer is not None:
        state.bias_by_layer = [None for _ in state.bias_by_layer]


def _auto_proto_frames_for_paper_budget(compress_frame_num: int, pq_modes: int) -> int:
    """Default K_max that preserves the paper's W : (K_max*S) ~= 1 : 3 split.

    Args:
        compress_frame_num: total visual frame budget |M| in frame units.
        pq_modes: S pseudo-frame modes decoded per prototype.
    """
    M = max(1, int(compress_frame_num))
    S = max(1, int(pq_modes))
    # Target far frames are approximately 3/4 of the total budget.
    target = int(round((0.75 * M) / float(S)))
    # Keep at least one near frame when possible.
    max_possible = max(1, (M - 1) // S)
    return max(1, min(max_possible, target))


def _resolve_proto_frames_max(prototrack_proto_frames: int, compress_frame_num: int, pq_modes: int) -> int:
    """Resolve K_max. Non-positive values mean paper-default automatic scaling."""
    if int(prototrack_proto_frames) > 0:
        return max(1, int(prototrack_proto_frames))
    return _auto_proto_frames_for_paper_budget(compress_frame_num, pq_modes)

def _get_or_reset_prototrack_state(model, num_layers, num_heads, max_proto_frames, device, reset) -> _ProtoTrackState:
    key = "_prototrack_kv_state"
    if reset or not hasattr(model, key):
        state = _new_state(num_layers, num_heads, max_proto_frames, device)
        setattr(model, key, state)
        return state
    state = getattr(model, key)
    if (
        not isinstance(state, _ProtoTrackState)
        or int(state.max_proto_frames) != int(max_proto_frames)
        or len(state.counts) != int(num_layers)
        or state.counts[0].shape[0] != int(num_heads)
    ):
        state = _new_state(num_layers, num_heads, max_proto_frames, device)
        setattr(model, key, state)
    return state


def _ensure_layer_aux_state(
    state: _ProtoTrackState,
    layer_idx: int,
    num_heads: int,
    max_proto_frames: int,
    token_per_frame: int,
    head_dim: int,
    pq_subspaces: int,
    pq_codebook_size: int,
    device,
    dtype=torch.float16,
) -> None:
    if state.mus[layer_idx] is None or state.mus[layer_idx].shape[:3] != (num_heads, max_proto_frames, token_per_frame):
        state.mus[layer_idx] = torch.zeros((num_heads, max_proto_frames, token_per_frame, 2), device=device, dtype=torch.float32)
        eye = torch.eye(2, device=device, dtype=torch.float32).view(1, 1, 1, 2, 2)
        state.sigmas[layer_idx] = eye.repeat(num_heads, max_proto_frames, token_per_frame, 1, 1).contiguous()

    if state.proto_keys is None:
        n = len(state.counts)
        state.proto_keys = [None for _ in range(n)]
        state.proto_values = [None for _ in range(n)]
    if (
        state.proto_keys[layer_idx] is None
        or state.proto_keys[layer_idx].shape != (num_heads, max_proto_frames, token_per_frame, head_dim)
        or state.proto_keys[layer_idx].device != device
    ):
        state.proto_keys[layer_idx] = torch.zeros((num_heads, max_proto_frames, token_per_frame, head_dim), device=device, dtype=dtype)
        state.proto_values[layer_idx] = torch.zeros((num_heads, max_proto_frames, token_per_frame, head_dim), device=device, dtype=dtype)
        state.counts[layer_idx].zero_()
        state.active[layer_idx].zero_()
        state.taus[layer_idx].zero_()
        state.initialized[layer_idx] = False
        state.proto_frames_cur[layer_idx] = 0

    if int(pq_subspaces) > 0 and int(pq_codebook_size) > 1:
        g_eff = _nearest_divisor(head_dim, int(pq_subspaces))
        c = int(pq_codebook_size)
        need_hist = (
            state.pq_hist_k is None
            or state.pq_hist_k[layer_idx] is None
            or state.pq_hist_k[layer_idx].shape != (num_heads, max_proto_frames, token_per_frame, g_eff, c)
        )
        if need_hist:
            if state.pq_hist_k is None:
                n = len(state.counts)
                state.pq_hist_k = [None for _ in range(n)]
                state.pq_hist_v = [None for _ in range(n)]
                state.pq_res_counts = [None for _ in range(n)]
                state.pq_codebooks_k = [None for _ in range(n)]
                state.pq_codebooks_v = [None for _ in range(n)]
            state.pq_hist_k[layer_idx] = torch.zeros((num_heads, max_proto_frames, token_per_frame, g_eff, c), device=device, dtype=torch.int32)
            state.pq_hist_v[layer_idx] = torch.zeros((num_heads, max_proto_frames, token_per_frame, g_eff, c), device=device, dtype=torch.int32)
            state.pq_res_counts[layer_idx] = torch.zeros((num_heads, max_proto_frames, token_per_frame), device=device, dtype=torch.int32)
            state.pq_codebooks_k[layer_idx] = None
            state.pq_codebooks_v[layer_idx] = None
            state.pq_hist_token_per_frame = int(token_per_frame)
            state.pq_last_subspaces = int(g_eff)
            state.pq_last_codebook_size = int(c)


# ---------------------------------------------------------------------------
# Geometry / PQ helpers
# ---------------------------------------------------------------------------

def _nearest_divisor(n: int, target: int) -> int:
    target = max(1, int(target))
    n = int(n)
    if n % target == 0:
        return target
    divs = [d for d in range(1, n + 1) if n % d == 0]
    divs.sort(key=lambda d: (abs(d - target), -d))
    return int(divs[0])


def _make_token_coords(token_per_frame: int, model=None, device=None) -> torch.Tensor:
    """Return normalized within-frame token coordinates [TPF,2]."""
    tpf = int(token_per_frame)
    side_h = None
    try:
        side_h = int(getattr(getattr(model, "config", None), "height_width"))
    except Exception:
        side_h = None
    if side_h is None or side_h <= 0:
        side_h = int(math.sqrt(tpf))
        side_h = max(1, side_h)
    side_w = int(math.ceil(tpf / side_h))
    ys = torch.arange(side_h, device=device, dtype=torch.float32)
    xs = torch.arange(side_w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1)[:tpf]
    if side_h > 1:
        coords[:, 0] = coords[:, 0] / float(side_h - 1)
    if side_w > 1:
        coords[:, 1] = coords[:, 1] / float(side_w - 1)
    return coords


@torch.no_grad()
def _pq_fit_codebooks(x: torch.Tensor, subspaces: int, codebook_size: int, iters: int = 4, sample_size: int = 4096, seed: int = 0) -> torch.Tensor:
    """Mini-batch k-means codebook fitting for a matrix [N,D]."""
    if x.numel() == 0:
        raise ValueError("Cannot fit PQ codebooks on empty residual set.")
    N, D = x.shape
    G = _nearest_divisor(D, int(subspaces))
    C = int(codebook_size)
    ds = D // G
    if N > int(sample_size):
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed))
        idx = torch.randperm(N, device=x.device, generator=gen)[: int(sample_size)]
        x_fit = x[idx]
    else:
        x_fit = x
    x_fit = x_fit.float()
    codebooks = torch.empty((G, C, ds), device=x.device, dtype=torch.float32)
    for g in range(G):
        xs = x_fit[:, g * ds : (g + 1) * ds]
        gen = torch.Generator(device=x.device)
        gen.manual_seed(int(seed + 1009 * (g + 1)))
        if xs.shape[0] >= C:
            centroids = xs[torch.randperm(xs.shape[0], device=x.device, generator=gen)[:C]].clone()
        else:
            repeat = int(math.ceil(C / max(1, xs.shape[0])))
            centroids = xs.repeat(repeat, 1)[:C].clone()
        for _ in range(int(iters)):
            dist = (xs.square().sum(1, keepdim=True) + centroids.square().sum(1).view(1, C) - 2.0 * (xs @ centroids.t()))
            assign = dist.argmin(dim=1)
            sums = torch.zeros_like(centroids)
            cnt = torch.zeros((C,), device=x.device, dtype=torch.float32)
            sums.index_add_(0, assign, xs)
            cnt.index_add_(0, assign, torch.ones((xs.shape[0],), device=x.device, dtype=torch.float32))
            empty = cnt <= 0
            if empty.any():
                repl = xs[torch.randperm(xs.shape[0], device=x.device, generator=gen)[: int(empty.sum().item())]]
                sums[empty] = repl
                cnt[empty] = 1.0
            centroids = sums / cnt.clamp(min=1.0).unsqueeze(1)
        codebooks[g] = centroids
    return codebooks.to(torch.float16)


@torch.no_grad()
def _pq_fit_codebooks_per_head(residuals: torch.Tensor, subspaces: int, codebook_size: int, iters: int, sample_size: int, seed: int) -> torch.Tensor:
    """Fit per-head codebooks. residuals: [H,N,D] -> [H,G,C,ds]."""
    H, N, D = residuals.shape
    cbs = []
    for h in range(H):
        cbs.append(_pq_fit_codebooks(residuals[h], subspaces, codebook_size, iters, sample_size, seed + 7919 * h))
    return torch.stack(cbs, dim=0)


@torch.no_grad()
def _pq_encode_indices_per_head(x: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """Encode x [H,N,D] with per-head codebooks [H,G,C,ds] -> [H,N,G]."""
    H, N, D = x.shape
    Hc, G, C, ds = codebooks.shape
    if H != Hc or D != G * ds:
        raise ValueError(f"PQ shape mismatch: x={tuple(x.shape)}, codebooks={tuple(codebooks.shape)}")
    codes = torch.empty((H, N, G), device=x.device, dtype=torch.long)
    xf = x.float()
    cb = codebooks.float()
    for h in range(H):
        for g in range(G):
            xs = xf[h, :, g * ds : (g + 1) * ds]
            cs = cb[h, g]
            dist = xs.square().sum(1, keepdim=True) + cs.square().sum(1).view(1, C) - 2.0 * (xs @ cs.t())
            codes[h, :, g] = dist.argmin(dim=1)
    return codes


@torch.no_grad()
def _decode_top_s_hist_batch(hist: torch.Tensor, codebooks: torch.Tensor, modes: int, beam_size: int, eps: float) -> torch.Tensor:
    """Algorithm 3 batched.

    hist: [H,P,TPF,G,C], codebooks: [H,G,C,ds]
    returns residual modes [H,P,TPF,S,D].
    """
    H, P, TPF, G, C = hist.shape
    _, Gc, Cc, ds = codebooks.shape
    assert G == Gc and C == Cc
    S_out = max(1, int(modes))
    B = int(beam_size) if int(beam_size) > 0 else 4 * S_out
    B = max(S_out, B)
    out_per_h = []
    for h in range(H):
        flat = hist[h].reshape(P * TPF, G, C).float()
        probs = (flat + float(eps)) / (flat.sum(dim=-1, keepdim=True) + C * float(eps)).clamp(min=float(eps))
        logp = torch.log(probs.clamp(min=1e-30))
        N = flat.shape[0]
        scores = torch.zeros((N, 1), device=hist.device, dtype=torch.float32)
        codes = torch.empty((N, 1, 0), device=hist.device, dtype=torch.long)
        cur_beam = 1
        for g in range(G):
            expanded = scores.unsqueeze(-1) + logp[:, g, :].unsqueeze(1)  # [N,cur_beam,C]
            expanded = expanded.reshape(N, cur_beam * C)
            keep = min(B, expanded.shape[1])
            new_scores, top_idx = torch.topk(expanded, k=keep, dim=1)
            prev_idx = top_idx // C
            code_idx = top_idx % C
            if codes.shape[-1] > 0:
                prev_codes = codes.gather(1, prev_idx.unsqueeze(-1).expand(-1, -1, codes.shape[-1]))
                codes = torch.cat([prev_codes, code_idx.unsqueeze(-1)], dim=-1)
            else:
                codes = code_idx.unsqueeze(-1)
            scores = new_scores
            cur_beam = keep
        keep_s = min(S_out, codes.shape[1])
        codes = codes[:, :keep_s, :]
        if keep_s < S_out:
            pad = codes[:, -1:, :].expand(-1, S_out - keep_s, -1)
            codes = torch.cat([codes, pad], dim=1)
        parts = []
        cb = codebooks[h].float()
        for g in range(G):
            idx = codes[:, :, g].reshape(-1)
            part = cb[g].index_select(0, idx).reshape(N, S_out, ds)
            parts.append(part)
        residual = torch.cat(parts, dim=-1).reshape(P, TPF, S_out, G * ds)
        out_per_h.append(residual)
    return torch.stack(out_per_h, dim=0)  # [H,P,TPF,S,D]



# ---------------------------------------------------------------------------
# RoPE re-anchoring helpers
# ---------------------------------------------------------------------------

def _get_rope_theta_from_model(model) -> float:
    """Best-effort extraction of the RoPE base used by the language model.

    ProtoKV keeps prototype Keys in a fixed rotary frame. When a new source
    frame at time t is absorbed into prototype k, we rotate the prototype key
    from its old anchor tau_k to t before the EMA update. This implements the
    paper's tau_k anchoring while remaining backend-independent. If the model
    config does not expose rope_theta, we fall back to the standard 10000.0.
    """
    seen = set()
    stack = [getattr(model, "config", None), getattr(getattr(model, "model", None), "config", None)]
    while stack:
        cfg = stack.pop()
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        val = getattr(cfg, "rope_theta", None)
        if val is not None:
            try:
                return float(val)
            except Exception:
                pass
        for name in ("text_config", "language_config", "llm_config", "vision_config"):
            child = getattr(cfg, name, None)
            if child is not None:
                stack.append(child)
    return 10000.0


def _rotate_half_split(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope_relative_rotate(x: torch.Tensor, delta_pos, rope_theta: Optional[float]) -> torch.Tensor:
    """Apply R(delta_pos) to already-rotated Keys.

    If x = R(src) x_base, this returns R(dst) x_base for delta_pos=dst-src.
    The implementation follows the split-half RoPE layout used by Qwen/Llama
    attention. For odd head dimensions or disabled theta, it is a no-op.
    """
    if rope_theta is None or x.shape[-1] % 2 != 0:
        return x
    D = x.shape[-1]
    device = x.device
    dtype = x.dtype
    inv_freq = 1.0 / (float(rope_theta) ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
    delta = torch.as_tensor(delta_pos, device=device, dtype=torch.float32)
    freqs = delta.unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    # Unsqueeze until emb broadcasts to x.
    while emb.dim() < x.dim():
        emb = emb.unsqueeze(-2)
    cos = emb.cos().to(dtype=dtype)
    sin = emb.sin().to(dtype=dtype)
    return (x * cos) + (_rotate_half_split(x) * sin)


def _reanchor_frame_key(frame_k: torch.Tensor, src_t: int, dst_t: int, rope_theta: Optional[float]) -> torch.Tensor:
    if rope_theta is None or int(src_t) == int(dst_t):
        return frame_k
    return _rope_relative_rotate(frame_k, int(dst_t) - int(src_t), rope_theta)


def _reanchor_proto_candidates(proto_k: torch.Tensor, taus: torch.Tensor, dst_t: int, rope_theta: Optional[float]) -> torch.Tensor:
    if rope_theta is None:
        return proto_k
    delta = int(dst_t) - taus.to(device=proto_k.device, dtype=torch.float32)
    return _rope_relative_rotate(proto_k, delta, rope_theta)

# ---------------------------------------------------------------------------
# Proto bank exact streaming update / maintenance
# ---------------------------------------------------------------------------

def _prototype_assignment_for_head(
    frame_k_h: torch.Tensor,
    proto_k_h: torch.Tensor,
    active_h: torch.Tensor,
    coords: torch.Tensor,
    mus_h: torch.Tensor,
    sigmas_h: torch.Tensor,
    taus_h: torch.Tensor,
    t: int,
    lambda_sp: float,
    lambda_idle: float,
    idle_threshold: int,
    rope_theta: Optional[float] = None,
) -> int:
    active_idx = torch.nonzero(active_h, as_tuple=False).flatten()
    if active_idx.numel() == 0:
        return -1
    pk = _reanchor_proto_candidates(proto_k_h[active_idx], taus_h[active_idx], int(t), rope_theta)  # [Pact,TPF,D]
    # Average token-wise cosine similarity.
    cos = F.cosine_similarity(frame_k_h.unsqueeze(0).float(), pk.float(), dim=-1).mean(dim=-1)  # [Pact]

    mu = mus_h[active_idx]      # [Pact,TPF,2]
    sigma = sigmas_h[active_idx].float()
    eye = torch.eye(2, device=frame_k_h.device, dtype=torch.float32).view(1, 1, 2, 2)
    sigma = sigma + 1e-4 * eye
    diff = (coords.view(1, -1, 2) - mu).unsqueeze(-1)  # [Pact,TPF,2,1]
    inv = torch.linalg.inv(sigma)
    mah = torch.matmul(torch.matmul(diff.transpose(-2, -1), inv), diff).squeeze(-1).squeeze(-1).mean(dim=-1)
    idle = (int(t) - taus_h[active_idx].long() > int(idle_threshold)).float()
    cost = -cos + float(lambda_sp) * mah + float(lambda_idle) * idle
    return int(active_idx[cost.argmin()].item())


@torch.no_grad()
def _init_proto_slot(
    proto_k: torch.Tensor,
    proto_v: torch.Tensor,
    counts: torch.Tensor,
    active: torch.Tensor,
    taus: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    hist_k: Optional[torch.Tensor],
    hist_v: Optional[torch.Tensor],
    res_counts: Optional[torch.Tensor],
    h: int,
    k: int,
    frame_k_h: torch.Tensor,
    frame_v_h: torch.Tensor,
    coords: torch.Tensor,
    t: int,
) -> None:
    proto_k[h, k].copy_(frame_k_h)
    proto_v[h, k].copy_(frame_v_h)
    counts[h, k] = 1.0
    active[h, k] = True
    taus[h, k] = int(t)
    mus[h, k].copy_(coords)
    eye = torch.eye(2, device=proto_k.device, dtype=torch.float32).view(1, 2, 2).expand(coords.shape[0], -1, -1)
    sigmas[h, k].copy_(eye)
    if hist_k is not None:
        hist_k[h, k].zero_()
    if hist_v is not None:
        hist_v[h, k].zero_()
    if res_counts is not None:
        res_counts[h, k].zero_()


@torch.no_grad()
def _update_residual_hist_one(
    hist_k: Optional[torch.Tensor],
    hist_v: Optional[torch.Tensor],
    res_counts: Optional[torch.Tensor],
    h: int,
    k: int,
    rK: torch.Tensor,  # [TPF,D]
    rV: torch.Tensor,
    codebook_k: Optional[torch.Tensor],  # [H,G,C,ds]
    codebook_v: Optional[torch.Tensor],
) -> None:
    if hist_k is None or hist_v is None or res_counts is None or codebook_k is None or codebook_v is None:
        return
    codes_k = _pq_encode_indices_per_head(rK.unsqueeze(0), codebook_k[h : h + 1]).squeeze(0)  # [TPF,G]
    codes_v = _pq_encode_indices_per_head(rV.unsqueeze(0), codebook_v[h : h + 1]).squeeze(0)
    TPF, G = codes_k.shape
    tok = torch.arange(TPF, device=rK.device)
    for g in range(G):
        hist_k[h, k, tok, g, codes_k[:, g]] += 1
        hist_v[h, k, tok, g, codes_v[:, g]] += 1
    res_counts[h, k] += 1


@torch.no_grad()
def _absorb_frame(
    frame_k: torch.Tensor,  # [H,TPF,D]
    frame_v: torch.Tensor,
    proto_k: torch.Tensor,
    proto_v: torch.Tensor,
    counts: torch.Tensor,
    active: torch.Tensor,
    taus: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    hist_k: Optional[torch.Tensor],
    hist_v: Optional[torch.Tensor],
    res_counts: Optional[torch.Tensor],
    codebook_k: Optional[torch.Tensor],
    codebook_v: Optional[torch.Tensor],
    coords: torch.Tensor,
    t: int,
    alpha: float,
    beta: float,
    eta: float,
    lambda_sp: float,
    lambda_idle: float,
    idle_threshold: int,
    rope_theta: Optional[float] = None,
    proto_update_ctx=None,
) -> None:
    H = frame_k.shape[0]
    for h in range(H):
        inactive = torch.nonzero(~active[h], as_tuple=False).flatten()
        if inactive.numel() > 0:
            k = int(inactive[0].item())
            _init_proto_slot(proto_k, proto_v, counts, active, taus, mus, sigmas, hist_k, hist_v, res_counts, h, k, frame_k[h], frame_v[h], coords, t)
            continue
        k = _prototype_assignment_for_head(
            frame_k[h], proto_k[h], active[h], coords, mus[h], sigmas[h], taus[h], t,
            lambda_sp=lambda_sp, lambda_idle=lambda_idle, idle_threshold=idle_threshold, rope_theta=rope_theta,
        )
        if k < 0:
            k = 0
        ctx = proto_update_ctx if proto_update_ctx is not None else _NullCtx(None)
        with ctx:
            # Keep prototype Keys anchored at the most recently absorbed source time tau_k.
            # Before absorbing a frame at time t, rotate the stored prototype from old tau_k
            # into the current frame's rotary frame, then set tau_k <- t.
            proto_k_current = _reanchor_frame_key(proto_k[h, k], int(taus[h, k].item()), int(t), rope_theta).float()
            proto_k[h, k] = ((1.0 - float(alpha)) * proto_k_current + float(alpha) * frame_k[h].float()).to(proto_k.dtype)
            proto_v[h, k] = ((1.0 - float(beta)) * proto_v[h, k].float() + float(beta) * frame_v[h].float()).to(proto_v.dtype)
            counts[h, k] += 1.0
            taus[h, k] = int(t)
            mu_new = (1.0 - float(eta)) * mus[h, k] + float(eta) * coords
            diff = (coords - mu_new).unsqueeze(-1)  # [TPF,2,1]
            outer = torch.matmul(diff, diff.transpose(-2, -1))
            sigmas[h, k] = (1.0 - float(eta)) * sigmas[h, k] + float(eta) * outer
            mus[h, k] = mu_new
        # Residuals are computed after center updates, matching Algorithm 1 order.
        rK = frame_k[h].float() - proto_k[h, k].float()
        rV = frame_v[h].float() - proto_v[h, k].float()
        _update_residual_hist_one(hist_k, hist_v, res_counts, h, k, rK, rV, codebook_k, codebook_v)


@torch.no_grad()
def _collect_residuals_for_codebook(
    frames_k: torch.Tensor,  # [H,F,TPF,D]
    frames_v: torch.Tensor,
    proto_k: torch.Tensor,
    proto_v: torch.Tensor,
    active: torch.Tensor,
    coords: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    taus: torch.Tensor,
    t0: int,
    lambda_sp: float,
    lambda_idle: float,
    idle_threshold: int,
    rope_theta: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    H, Fnum, TPF, D = frames_k.shape
    residuals_k = [[] for _ in range(H)]
    residuals_v = [[] for _ in range(H)]
    for f in range(Fnum):
        t = int(t0 + f)
        for h in range(H):
            k = _prototype_assignment_for_head(frames_k[h, f], proto_k[h], active[h], coords, mus[h], sigmas[h], taus[h], t, lambda_sp, lambda_idle, idle_threshold, rope_theta=rope_theta)
            if k < 0:
                k = 0
            residuals_k[h].append((frames_k[h, f].float() - _reanchor_frame_key(proto_k[h, k], int(taus[h, k].item()), t, rope_theta).float()).reshape(TPF, D))
            residuals_v[h].append((frames_v[h, f].float() - proto_v[h, k].float()).reshape(TPF, D))
    rk = torch.stack([torch.cat(residuals_k[h], dim=0) if residuals_k[h] else torch.zeros((1, D), device=frames_k.device) for h in range(H)], dim=0)
    rv = torch.stack([torch.cat(residuals_v[h], dim=0) if residuals_v[h] else torch.zeros((1, D), device=frames_v.device) for h in range(H)], dim=0)
    return rk, rv


@torch.no_grad()
def _rebuild_histograms_from_frames(
    frames_k: torch.Tensor,
    frames_v: torch.Tensor,
    proto_k: torch.Tensor,
    proto_v: torch.Tensor,
    counts: torch.Tensor,
    active: torch.Tensor,
    taus: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    hist_k: torch.Tensor,
    hist_v: torch.Tensor,
    res_counts: torch.Tensor,
    codebook_k: torch.Tensor,
    codebook_v: torch.Tensor,
    coords: torch.Tensor,
    t0: int,
    lambda_sp: float,
    lambda_idle: float,
    idle_threshold: int,
    rope_theta: Optional[float] = None,
) -> None:
    hist_k.zero_(); hist_v.zero_(); res_counts.zero_()
    H, Fnum, _, _ = frames_k.shape
    for f in range(Fnum):
        t = int(t0 + f)
        for h in range(H):
            k = _prototype_assignment_for_head(frames_k[h, f], proto_k[h], active[h], coords, mus[h], sigmas[h], taus[h], t, lambda_sp, lambda_idle, idle_threshold, rope_theta=rope_theta)
            if k < 0:
                continue
            rK = frames_k[h, f].float() - _reanchor_frame_key(proto_k[h, k], int(taus[h, k].item()), t, rope_theta).float()
            rV = frames_v[h, f].float() - proto_v[h, k].float()
            _update_residual_hist_one(hist_k, hist_v, res_counts, h, k, rK, rV, codebook_k, codebook_v)


@torch.no_grad()
def _merge_slots_for_head(
    proto_k, proto_v, counts, active, taus, mus, sigmas, hist_k, hist_v, res_counts, h: int, i: int, j: int, rope_theta: Optional[float] = None) -> None:
    ni = counts[h, i].clone()
    nj = counts[h, j].clone()
    denom = (ni + nj).clamp(min=1.0)
    wi = (ni / denom).to(torch.float32)
    wj = (nj / denom).to(torch.float32)
    tau_new = int(torch.maximum(taus[h, i], taus[h, j]).item())
    key_i = _reanchor_frame_key(proto_k[h, i], int(taus[h, i].item()), tau_new, rope_theta).float()
    key_j = _reanchor_frame_key(proto_k[h, j], int(taus[h, j].item()), tau_new, rope_theta).float()
    proto_k[h, i] = (wi * key_i + wj * key_j).to(proto_k.dtype)
    proto_v[h, i] = (wi * proto_v[h, i].float() + wj * proto_v[h, j].float()).to(proto_v.dtype)
    mus[h, i] = wi * mus[h, i] + wj * mus[h, j]
    sigmas[h, i] = wi * sigmas[h, i] + wj * sigmas[h, j]
    counts[h, i] = ni + nj
    taus[h, i] = tau_new
    if hist_k is not None:
        hist_k[h, i] += hist_k[h, j]
        hist_k[h, j].zero_()
    if hist_v is not None:
        hist_v[h, i] += hist_v[h, j]
        hist_v[h, j].zero_()
    if res_counts is not None:
        res_counts[h, i] += res_counts[h, j]
        res_counts[h, j].zero_()
    counts[h, j] = 0.0
    active[h, j] = False


@torch.no_grad()
def _prototype_maintenance(
    proto_k: torch.Tensor,
    proto_v: torch.Tensor,
    counts: torch.Tensor,
    active: torch.Tensor,
    taus: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    hist_k: Optional[torch.Tensor],
    hist_v: Optional[torch.Tensor],
    res_counts: Optional[torch.Tensor],
    near_k: torch.Tensor,
    near_v: torch.Tensor,
    coords: torch.Tensor,
    t: int,
    idle_threshold: int,
    decay_gamma: float,
    merge_eps_k: float,
    merge_eps_v: float,
    n_min: float,
    rope_theta: Optional[float] = None,
) -> None:
    H, P, TPF, D = proto_k.shape
    # Idle decay.
    idle = (int(t) - taus.long()) > int(idle_threshold)
    # Paper Algorithm 2 uses an idle-decay *rate* gamma: n_k <- floor((1-gamma) n_k).
    retention = max(0.0, min(1.0, 1.0 - float(decay_gamma)))
    counts[idle & active] = torch.floor(retention * counts[idle & active])
    active[counts < float(n_min)] = False

    # Merge close prototypes, per head.
    for h in range(H):
        for i in range(P):
            if not bool(active[h, i]):
                continue
            for j in range(i + 1, P):
                if not bool(active[h, j]):
                    continue
                tau_cmp = int(torch.maximum(taus[h, i], taus[h, j]).item())
                key_i_cmp = _reanchor_frame_key(proto_k[h, i], int(taus[h, i].item()), tau_cmp, rope_theta).float()
                key_j_cmp = _reanchor_frame_key(proto_k[h, j], int(taus[h, j].item()), tau_cmp, rope_theta).float()
                dk = torch.norm((key_i_cmp - key_j_cmp).reshape(-1)) / math.sqrt(float(TPF * D))
                dv = torch.norm((proto_v[h, i].float() - proto_v[h, j].float()).reshape(-1)) / math.sqrt(float(TPF * D))
                if float(dk.item()) < float(merge_eps_k) and float(dv.item()) < float(merge_eps_v):
                    _merge_slots_for_head(proto_k, proto_v, counts, active, taus, mus, sigmas, hist_k, hist_v, res_counts, h, i, j, rope_theta=rope_theta)

    # Recycle inactive / low mass slots from recent exact window.
    if near_k.numel() == 0:
        return
    # Select most recent frame from B_near.
    kr = near_k[:, -1]
    vr = near_v[:, -1]
    for h in range(H):
        for k in range(P):
            if (not bool(active[h, k])) or float(counts[h, k].item()) < float(n_min):
                _init_proto_slot(proto_k, proto_v, counts, active, taus, mus, sigmas, hist_k, hist_v, res_counts, h, k, kr[h], vr[h], coords, t)


# ---------------------------------------------------------------------------
# Compression layer
# ---------------------------------------------------------------------------

@torch.no_grad()
def _protokv_compress_layer_exact(
    key_states_to_compress: torch.Tensor,
    value_states_to_compress: torch.Tensor,
    token_per_frame: int,
    compress_frame_num: int,
    proto_frames_max: int,
    counts: torch.Tensor,
    active: torch.Tensor,
    taus: torch.Tensor,
    mus: torch.Tensor,
    sigmas: torch.Tensor,
    proto_bank_k: torch.Tensor,
    proto_bank_v: torch.Tensor,
    state_initialized: bool,
    state_proto_frames_cur: int,
    cur_frame_end: Optional[int],
    coords: torch.Tensor,
    pq_subspaces: int,
    pq_codebook_size: int,
    pq_kmeans_iters: int,
    pq_sample_size: int,
    pq_codebook_k: Optional[torch.Tensor],
    pq_codebook_v: Optional[torch.Tensor],
    pq_hist_k: Optional[torch.Tensor],
    pq_hist_v: Optional[torch.Tensor],
    pq_res_counts: Optional[torch.Tensor],
    pq_seed: int,
    pq_modes: int,
    pq_beam_size: int,
    pq_beam_eps: float,
    lambda_sp: float,
    lambda_idle: float,
    idle_threshold: int,
    alpha: float,
    beta: float,
    eta: float,
    maintenance_gamma: float,
    merge_eps_k: float,
    merge_eps_v: float,
    n_min: float,
    cuda_timer=None,
    timing: Optional[Dict[str, float]] = None,
    rope_theta: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    assert key_states_to_compress.dim() == 4 and key_states_to_compress.shape[0] == 1
    _, H, L, D = key_states_to_compress.shape
    assert L % int(token_per_frame) == 0
    cur_frames = L // int(token_per_frame)
    S_modes = max(1, int(pq_modes))

    # For a fixed output budget in frame-units: L_context = W + Kmax*S.
    proto_frames_target = min(int(proto_frames_max), max(1, (int(compress_frame_num) - 1) // S_modes))
    recent_frames = int(compress_frame_num) - proto_frames_target * S_modes
    if recent_frames < 1:
        proto_frames_target = max(1, (int(compress_frame_num) - 1) // S_modes)
        recent_frames = int(compress_frame_num) - proto_frames_target * S_modes
    if cur_frames <= int(compress_frame_num):
        bias = torch.zeros((1, H, L), device=key_states_to_compress.device, dtype=torch.float32)
        return key_states_to_compress, value_states_to_compress, bias, state_initialized, int(state_proto_frames_cur), pq_codebook_k, pq_codebook_v

    kf = key_states_to_compress[0].reshape(H, cur_frames, int(token_per_frame), D)
    vf = value_states_to_compress[0].reshape(H, cur_frames, int(token_per_frame), D)
    device = kf.device
    t_end = int(cur_frame_end) if cur_frame_end is not None else cur_frames
    t_start = t_end - cur_frames

    near_start = cur_frames - recent_frames
    near_k = kf[:, near_start:].contiguous()
    near_v = vf[:, near_start:].contiguous()

    def ctx_factory():
        if cuda_timer is not None:
            return cuda_timer("proto_update")
        return _NullCtx(timing)

    # Persistent centers c_k,p_k live in state.proto_keys/proto_values.
    # We never recover them from decoded pseudo-token cache outputs.
    if proto_bank_k.shape != (H, int(proto_frames_max), int(token_per_frame), D):
        raise ValueError(f"Bad proto_bank_k shape {tuple(proto_bank_k.shape)}")
    if not state_initialized or int(state_proto_frames_cur) <= 0:
        far_frames = max(0, near_start)
        proto_k = proto_bank_k[:, :proto_frames_target]
        proto_v = proto_bank_v[:, :proto_frames_target]
        proto_k.zero_(); proto_v.zero_()
        # Reset state slots.
        counts[:, :proto_frames_target].zero_(); active[:, :proto_frames_target].zero_(); taus[:, :proto_frames_target].zero_()
        if pq_hist_k is not None:
            pq_hist_k[:, :proto_frames_target].zero_(); pq_hist_v[:, :proto_frames_target].zero_(); pq_res_counts[:, :proto_frames_target].zero_()
        if far_frames > 0:
            # Algorithm 1 initialization: fill inactive slots first with earliest evicted frames.
            for f in range(far_frames):
                _absorb_frame(
                    kf[:, f], vf[:, f], proto_k, proto_v, counts[:, :proto_frames_target], active[:, :proto_frames_target], taus[:, :proto_frames_target],
                    mus[:, :proto_frames_target], sigmas[:, :proto_frames_target],
                    None, None, None, None, None, coords, int(t_start + f), alpha, beta, eta, lambda_sp, lambda_idle, idle_threshold,
                    rope_theta=rope_theta, proto_update_ctx=ctx_factory(),
                )
                if bool(active[:, :proto_frames_target].all()):
                    _prototype_maintenance(
                        proto_k, proto_v, counts[:, :proto_frames_target], active[:, :proto_frames_target], taus[:, :proto_frames_target],
                        mus[:, :proto_frames_target], sigmas[:, :proto_frames_target], None, None, None,
                        near_k, near_v, coords, int(t_start + f), idle_threshold, maintenance_gamma, merge_eps_k, merge_eps_v, n_min, rope_theta=rope_theta,
                    )
        else:
            # No far frames yet; seed from near frame to maintain a valid bank.
            seed = 0
            for h in range(H):
                _init_proto_slot(proto_k, proto_v, counts[:, :proto_frames_target], active[:, :proto_frames_target], taus[:, :proto_frames_target], mus[:, :proto_frames_target], sigmas[:, :proto_frames_target], None, None, None, h, seed, near_k[h, -1], near_v[h, -1], coords, t_end)
        state_initialized = True
        state_proto_frames_cur = proto_frames_target

        # Fit PQ codebooks and rebuild histograms from far frames once prototypes exist.
        if int(pq_subspaces) > 0 and pq_hist_k is not None and far_frames > 0:
            rk, rv = _collect_residuals_for_codebook(kf[:, :far_frames], vf[:, :far_frames], proto_k, proto_v, active[:, :proto_frames_target], coords, mus[:, :proto_frames_target], sigmas[:, :proto_frames_target], taus[:, :proto_frames_target], t_start, lambda_sp, lambda_idle, idle_threshold, rope_theta=rope_theta)
            pq_codebook_k = _pq_fit_codebooks_per_head(rk, pq_subspaces, pq_codebook_size, pq_kmeans_iters, pq_sample_size, pq_seed + 17)
            pq_codebook_v = _pq_fit_codebooks_per_head(rv, pq_subspaces, pq_codebook_size, pq_kmeans_iters, pq_sample_size, pq_seed + 29)
            _rebuild_histograms_from_frames(kf[:, :far_frames], vf[:, :far_frames], proto_k, proto_v, counts[:, :proto_frames_target], active[:, :proto_frames_target], taus[:, :proto_frames_target], mus[:, :proto_frames_target], sigmas[:, :proto_frames_target], pq_hist_k[:, :proto_frames_target], pq_hist_v[:, :proto_frames_target], pq_res_counts[:, :proto_frames_target], pq_codebook_k, pq_codebook_v, coords, t_start, lambda_sp, lambda_idle, idle_threshold, rope_theta=rope_theta)
    else:
        P_old = int(state_proto_frames_cur)
        P_old = min(P_old, counts.shape[1], max(1, P_old))
        proto_k = proto_bank_k[:, :max(P_old, proto_frames_target)]
        proto_v = proto_bank_v[:, :max(P_old, proto_frames_target)]

        # Resize target capacity if needed.
        if proto_frames_target != P_old:
            P_new = proto_frames_target
            if P_new < P_old:
                # Merge removed slots into closest kept slots, including histograms.
                for h in range(H):
                    for j in range(P_new, P_old):
                        if not bool(active[h, j]):
                            continue
                        sim = F.cosine_similarity(proto_k[h, j].mean(0).float().view(1, -1), proto_k[h, :P_new].mean(1).float(), dim=-1)
                        i = int(sim.argmax().item())
                        _merge_slots_for_head(proto_k, proto_v, counts, active, taus, mus, sigmas, pq_hist_k, pq_hist_v, pq_res_counts, h, i, j, rope_theta=rope_theta)
                active[:, P_new:P_old] = False; counts[:, P_new:P_old] = 0
                if pq_hist_k is not None:
                    pq_hist_k[:, P_new:P_old].zero_(); pq_hist_v[:, P_new:P_old].zero_(); pq_res_counts[:, P_new:P_old].zero_()
                P_old = P_new
            elif P_new > P_old:
                # New slots are inactive until Algorithm 1 initializes them from evicted frames.
                active[:, P_old:P_new] = False; counts[:, P_old:P_new] = 0
                proto_k[:, P_old:P_new].zero_(); proto_v[:, P_old:P_new].zero_()
                if pq_hist_k is not None:
                    pq_hist_k[:, P_old:P_new].zero_(); pq_hist_v[:, P_old:P_new].zero_(); pq_res_counts[:, P_old:P_new].zero_()
                P_old = P_new
            proto_k = proto_bank_k[:, :P_old]
            proto_v = proto_bank_v[:, :P_old]
            state_proto_frames_cur = P_old

        P_cur = proto_frames_target
        # Exact frames to absorb are everything after the previous pseudo-prefix and before new near window.
        old_pseudo_frames = min(cur_frames, int(state_proto_frames_cur) * S_modes)
        absorb_start = old_pseudo_frames if state_initialized else 0
        absorb_end = near_start
        if absorb_end > absorb_start:
            for f in range(absorb_start, absorb_end):
                _absorb_frame(
                    kf[:, f], vf[:, f], proto_k, proto_v, counts[:, :P_cur], active[:, :P_cur], taus[:, :P_cur],
                    mus[:, :P_cur], sigmas[:, :P_cur], pq_hist_k[:, :P_cur] if pq_hist_k is not None else None, pq_hist_v[:, :P_cur] if pq_hist_v is not None else None, pq_res_counts[:, :P_cur] if pq_res_counts is not None else None,
                    pq_codebook_k, pq_codebook_v, coords, int(t_start + f), alpha, beta, eta, lambda_sp, lambda_idle, idle_threshold,
                    rope_theta=rope_theta, proto_update_ctx=ctx_factory(),
                )
                if bool(active[:, :P_cur].all()):
                    _prototype_maintenance(
                        proto_k, proto_v, counts[:, :P_cur], active[:, :P_cur], taus[:, :P_cur],
                        mus[:, :P_cur], sigmas[:, :P_cur],
                        pq_hist_k[:, :P_cur] if pq_hist_k is not None else None,
                        pq_hist_v[:, :P_cur] if pq_hist_v is not None else None,
                        pq_res_counts[:, :P_cur] if pq_res_counts is not None else None,
                        near_k, near_v, coords, int(t_start + f), idle_threshold, maintenance_gamma, merge_eps_k, merge_eps_v, n_min, rope_theta=rope_theta,
                    )

        # If codebooks do not exist yet, fit from absorbed exact frames and rebuild hist.
        if int(pq_subspaces) > 0 and pq_hist_k is not None and (pq_codebook_k is None or pq_codebook_v is None) and near_start > 0:
            far_for_fit_start = old_pseudo_frames if state_initialized and old_pseudo_frames < near_start else 0
            fit_k = kf[:, far_for_fit_start:near_start]
            fit_v = vf[:, far_for_fit_start:near_start]
            if fit_k.shape[1] > 0:
                rk, rv = _collect_residuals_for_codebook(fit_k, fit_v, proto_k, proto_v, active[:, :P_cur], coords, mus[:, :P_cur], sigmas[:, :P_cur], taus[:, :P_cur], t_start + far_for_fit_start, lambda_sp, lambda_idle, idle_threshold, rope_theta=rope_theta)
                pq_codebook_k = _pq_fit_codebooks_per_head(rk, pq_subspaces, pq_codebook_size, pq_kmeans_iters, pq_sample_size, pq_seed + 17)
                pq_codebook_v = _pq_fit_codebooks_per_head(rv, pq_subspaces, pq_codebook_size, pq_kmeans_iters, pq_sample_size, pq_seed + 29)
                _rebuild_histograms_from_frames(fit_k, fit_v, proto_k, proto_v, counts[:, :P_cur], active[:, :P_cur], taus[:, :P_cur], mus[:, :P_cur], sigmas[:, :P_cur], pq_hist_k[:, :P_cur], pq_hist_v[:, :P_cur], pq_res_counts[:, :P_cur], pq_codebook_k, pq_codebook_v, coords, t_start + far_for_fit_start, lambda_sp, lambda_idle, idle_threshold, rope_theta=rope_theta)

    P_final = proto_frames_target
    # Algorithm 2 maintenance is applied immediately after each absorbed frame once
    # the prototype bank is populated, matching the streaming update flow.

    # Algorithm 4: top-S pseudo-token synthesis.
    proto_base_k = proto_k[:, :P_final].contiguous()
    proto_base_v = proto_v[:, :P_final].contiguous()
    if int(pq_subspaces) > 0 and pq_hist_k is not None and pq_codebook_k is not None and pq_codebook_v is not None:
        rK_modes = _decode_top_s_hist_batch(pq_hist_k[:, :P_final], pq_codebook_k, S_modes, pq_beam_size, pq_beam_eps).to(proto_base_k.dtype)
        rV_modes = _decode_top_s_hist_batch(pq_hist_v[:, :P_final], pq_codebook_v, S_modes, pq_beam_size, pq_beam_eps).to(proto_base_v.dtype)
        # If n_res == 0, residual modes are zero by Algorithm 4.
        if pq_res_counts is not None:
            has_res = (pq_res_counts[:, :P_final] > 0).to(proto_base_k.dtype)[..., None, None]
            rK_modes = rK_modes * has_res
            rV_modes = rV_modes * has_res
        proto_modes_k = proto_base_k.unsqueeze(3) + rK_modes  # [H,P,TPF,S,D]
        proto_modes_v = proto_base_v.unsqueeze(3) + rV_modes
    else:
        proto_modes_k = proto_base_k.unsqueeze(3).expand(-1, -1, -1, S_modes, -1)
        proto_modes_v = proto_base_v.unsqueeze(3).expand(-1, -1, -1, S_modes, -1)

    # Prototype Keys are stored in their latest tau_k rotary frame; all S modes from
    # prototype k inherit that anchor. Order: prototype k, modes s, then within-frame tokens.
    proto_out_k = proto_modes_k.permute(0, 1, 3, 2, 4).reshape(H, P_final * S_modes, int(token_per_frame), D).contiguous()
    proto_out_v = proto_modes_v.permute(0, 1, 3, 2, 4).reshape(H, P_final * S_modes, int(token_per_frame), D).contiguous()
    new_kf = torch.cat([proto_out_k, near_k], dim=1)
    new_vf = torch.cat([proto_out_v, near_v], dim=1)
    assert new_kf.shape[1] == int(compress_frame_num), (new_kf.shape, compress_frame_num, P_final, S_modes, recent_frames)
    new_k = new_kf.reshape(1, H, int(compress_frame_num) * int(token_per_frame), D)
    new_v = new_vf.reshape(1, H, int(compress_frame_num) * int(token_per_frame), D)

    # Bias vector b: zero for near tokens; log n_k for every pseudo-token from prototype k.
    log_counts = torch.log(counts[:, :P_final].clamp(min=1.0)).to(torch.float32)  # [H,P]
    proto_bias = log_counts[:, :, None, None].expand(H, P_final, S_modes, int(token_per_frame)).reshape(H, P_final * S_modes * int(token_per_frame))
    near_bias = torch.zeros((H, recent_frames * int(token_per_frame)), device=device, dtype=torch.float32)
    bias_vis = torch.cat([proto_bias, near_bias], dim=-1).unsqueeze(0)  # [1,H,Lvis]
    return new_k, new_v, bias_vis, state_initialized, P_final, pq_codebook_k, pq_codebook_v


# ---------------------------------------------------------------------------
# Attention bias hook
# ---------------------------------------------------------------------------

def _get_language_layers(model) -> Optional[List[Any]]:
    for path in (
        ("model", "language_model", "layers"),
        ("model", "model", "layers"),
        ("model", "layers"),
        ("language_model", "layers"),
    ):
        obj = model
        ok = True
        for name in path:
            if not hasattr(obj, name):
                ok = False
                break
            obj = getattr(obj, name)
        if ok:
            try:
                return list(obj)
            except Exception:
                return obj
    return None


def install_protokv_attention_bias_hook(model) -> None:
    """Patch language self-attention modules to add per-layer ProtoKV logit bias.

    This wrapper modifies the `attention_mask` passed into each attention layer. It
    requires an additive/eager attention path. For Qwen, load with
    `attn_implementation="eager"` when using the exact log-mass bias.
    """
    if getattr(model, "_protokv_bias_hook_installed", False):
        return
    layers = _get_language_layers(model)
    if layers is None:
        return
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None or getattr(attn, "_protokv_wrapped", False) or not hasattr(attn, "forward"):
            continue
        orig_forward = attn.forward
        sig = inspect.signature(orig_forward)
        params = list(sig.parameters.keys())

        def make_wrapped(orig, layer_i: int, param_names: List[str]):
            def wrapped(*args, **kwargs):
                bias_layers = getattr(model, "_protokv_bias_by_layer", None)
                if bias_layers is None or layer_i >= len(bias_layers) or bias_layers[layer_i] is None:
                    return orig(*args, **kwargs)
                # Locate attention_mask in args/kwargs.
                attn_mask = kwargs.get("attention_mask", None)
                args_list = list(args)
                mask_pos = None
                if "attention_mask" in param_names:
                    mask_pos = param_names.index("attention_mask")
                    if attn_mask is None and mask_pos < len(args_list):
                        attn_mask = args_list[mask_pos]
                # Bias has shape [1,H,K_bias]. Add to the first K_bias key positions;
                # current prompt/generated tokens and normal near/system tokens receive zero.
                bias = bias_layers[layer_i]
                if not torch.is_tensor(bias):
                    return orig(*args, **kwargs)

                # Locate hidden_states so we can build an additive mask when the
                # model does not pass one into the attention layer. This happens for
                # both the first question-token block (q_len > 1) and autoregressive
                # decoding (q_len = 1) in some Qwen/Transformers versions.
                hidden_states = kwargs.get("hidden_states", None)
                if hidden_states is None and "hidden_states" in param_names:
                    hs_pos = param_names.index("hidden_states")
                    if hs_pos < len(args_list):
                        hidden_states = args_list[hs_pos]

                def _infer_total_k_len(q_len: int, default_len: int) -> int:
                    pkv = kwargs.get("past_key_value", None)
                    if pkv is None and "past_key_value" in param_names:
                        pkv_pos = param_names.index("past_key_value")
                        if pkv_pos < len(args_list):
                            pkv = args_list[pkv_pos]
                    past_len = None
                    if pkv is not None:
                        try:
                            past_len = int(pkv.get_seq_length(layer_i))
                        except Exception:
                            try:
                                past_len = int(pkv.get_seq_length())
                            except Exception:
                                past_len = None
                        if past_len is None:
                            try:
                                layers_obj = getattr(pkv, "layers", None)
                                if layers_obj is not None and layer_i < len(layers_obj):
                                    keys = getattr(layers_obj[layer_i], "keys", None)
                                    if torch.is_tensor(keys):
                                        past_len = int(keys.shape[-2])
                            except Exception:
                                past_len = None
                        if past_len is None and isinstance(pkv, (tuple, list)) and len(pkv) > 0:
                            try:
                                keys = pkv[0]
                                if torch.is_tensor(keys):
                                    past_len = int(keys.shape[-2])
                            except Exception:
                                past_len = None
                    if past_len is not None:
                        return max(default_len, past_len + int(q_len))
                    return default_len

                if attn_mask is None:
                    if hidden_states is None or not torch.is_tensor(hidden_states):
                        return orig(*args, **kwargs)
                    q_len_tmp = int(hidden_states.shape[-2])
                    btmp = bias.to(device=hidden_states.device, dtype=hidden_states.dtype)
                    default_k_len = int(btmp.shape[-1]) + q_len_tmp
                    k_len_tmp = _infer_total_k_len(q_len_tmp, default_k_len)
                    attn_mask = torch.zeros(
                        (hidden_states.shape[0], btmp.shape[1], q_len_tmp, k_len_tmp),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )

                if attn_mask.dim() < 4:
                    # The model should normally convert 2D masks to 4D before the layer.
                    return orig(*args, **kwargs)
                Bsz, mask_heads, q_len, k_len = attn_mask.shape
                b = bias.to(device=attn_mask.device, dtype=attn_mask.dtype)
                if b.dim() == 3:
                    # [1,H,K]
                    h_bias = b.shape[1]
                    kb = min(int(b.shape[-1]), int(k_len))
                    add = torch.zeros((1, h_bias, 1, k_len), device=attn_mask.device, dtype=attn_mask.dtype)
                    add[..., :kb] = b[..., :kb].unsqueeze(-2)
                    # Preserve the paper's per-head log-mass bias. If the model provides
                    # a single-head additive mask, expand it across heads rather than averaging bias.
                    if mask_heads == 1 and h_bias != 1:
                        attn_mask = attn_mask.expand(Bsz, h_bias, q_len, k_len)
                    elif mask_heads != h_bias:
                        add = add[:, :1]
                    new_mask = attn_mask + add
                    if mask_pos is not None and mask_pos < len(args_list):
                        args_list[mask_pos] = new_mask
                    else:
                        kwargs["attention_mask"] = new_mask
                return orig(*tuple(args_list), **kwargs)
            return wrapped

        attn.forward = make_wrapped(orig_forward, layer_idx, params)
        attn._protokv_wrapped = True
    model._protokv_bias_hook_installed = True


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_kv_cache(
    past_key_values,
    model,
    system_size: int,
    inst_size: int,
    token_per_frame: int,
    compress_frame_num: int,
    method: str = "uniform",
    tar_ratio: float = 0.5,
    query_ratio: float = 0.25,
    adaptive_pooling: bool = False,
    is_first_block: bool = False,
    is_last_block: bool = False,
    per_frame: bool = False,
    prototrack_proto_frames: int = 0,
    prototrack_frame_end: Optional[int] = None,
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
    cuda_timer=None,
    timing: Optional[Dict[str, float]] = None,
) -> Tuple:
    if method != "prototrack-kv":
        clear_protokv_attention_bias(model)
    if compress_frame_num <= 0:
        if method == "prototrack-kv":
            clear_protokv_attention_bias(model)
        return past_key_values, None

    current_seq_len = past_key_values.get_seq_length()
    vision_start = int(system_size)
    vision_end = current_seq_len
    vision_length = vision_end - vision_start
    assert vision_length % int(token_per_frame) == 0, (vision_length, token_per_frame)
    current_frame_num = vision_length // int(token_per_frame)
    if current_frame_num <= int(compress_frame_num):
        if method == "prototrack-kv":
            clear_protokv_attention_bias(model)
        return past_key_values, None

    # Get language layers robustly.
    layers = _get_language_layers(model)
    if layers is None:
        # Fall back to the path used by Qwen2.5-VL.
        layers = model.model.language_model.layers
    num_layers = len(layers)

    if method == "prototrack-kv":
        ks0 = past_key_values.layers[0].keys
        device0 = ks0.device
        _, num_heads0, _, _ = ks0.shape
        proto_frames_max = _resolve_proto_frames_max(prototrack_proto_frames, compress_frame_num, prototrack_pq_modes)
        state = _get_or_reset_prototrack_state(model, num_layers, num_heads0, proto_frames_max, device0, reset=bool(is_first_block))
        rope_theta = _get_rope_theta_from_model(model)
        install_protokv_attention_bias_hook(model)

    for layer_idx in range(num_layers):
        key_states = past_key_values.layers[layer_idx].keys
        value_states = past_key_values.layers[layer_idx].values
        key_states_to_compress = key_states[:, :, system_size:, :]
        value_states_to_compress = value_states[:, :, system_size:, :]
        _, num_heads, _, head_dim = key_states_to_compress.shape

        if method == "prototrack-kv":
            _ensure_layer_aux_state(
                state, layer_idx, num_heads, state.max_proto_frames, int(token_per_frame), int(head_dim),
                int(prototrack_pq_subspaces), int(prototrack_pq_codebook_size), key_states.device, key_states.dtype,
            )
            coords = _make_token_coords(int(token_per_frame), model=model, device=key_states.device)
            S_modes = max(1, int(prototrack_pq_modes))
            beam = int(prototrack_pq_beam_size) if int(prototrack_pq_beam_size) > 0 else 4 * S_modes
            new_k_vis, new_v_vis, bias_vis, state.initialized[layer_idx], p_final, cb_k, cb_v = _protokv_compress_layer_exact(
                key_states_to_compress=key_states_to_compress,
                value_states_to_compress=value_states_to_compress,
                token_per_frame=int(token_per_frame),
                compress_frame_num=int(compress_frame_num),
                proto_frames_max=state.max_proto_frames,
                counts=state.counts[layer_idx],
                active=state.active[layer_idx],
                taus=state.taus[layer_idx],
                mus=state.mus[layer_idx],
                sigmas=state.sigmas[layer_idx],
                proto_bank_k=state.proto_keys[layer_idx],
                proto_bank_v=state.proto_values[layer_idx],
                state_initialized=bool(state.initialized[layer_idx]),
                state_proto_frames_cur=int(state.proto_frames_cur[layer_idx]),
                cur_frame_end=prototrack_frame_end,
                coords=coords,
                pq_subspaces=int(prototrack_pq_subspaces),
                pq_codebook_size=int(prototrack_pq_codebook_size),
                pq_kmeans_iters=int(prototrack_pq_kmeans_iters),
                pq_sample_size=int(prototrack_pq_sample_size),
                pq_codebook_k=state.pq_codebooks_k[layer_idx] if state.pq_codebooks_k is not None else None,
                pq_codebook_v=state.pq_codebooks_v[layer_idx] if state.pq_codebooks_v is not None else None,
                pq_hist_k=state.pq_hist_k[layer_idx] if state.pq_hist_k is not None else None,
                pq_hist_v=state.pq_hist_v[layer_idx] if state.pq_hist_v is not None else None,
                pq_res_counts=state.pq_res_counts[layer_idx] if state.pq_res_counts is not None else None,
                pq_seed=int(prototrack_pq_seed + layer_idx * 101),
                pq_modes=S_modes,
                pq_beam_size=beam,
                pq_beam_eps=float(prototrack_pq_beam_eps),
                lambda_sp=float(prototrack_lambda_sp),
                lambda_idle=float(prototrack_lambda_idle),
                idle_threshold=int(prototrack_idle_threshold),
                alpha=float(prototrack_alpha),
                beta=float(prototrack_beta),
                eta=float(prototrack_eta),
                maintenance_gamma=float(prototrack_maintenance_gamma),
                merge_eps_k=float(prototrack_merge_eps_k),
                merge_eps_v=float(prototrack_merge_eps_v),
                n_min=float(prototrack_min_mass),
                cuda_timer=cuda_timer,
                timing=timing if cuda_timer is None else None,
                rope_theta=rope_theta,
            )
            state.proto_frames_cur[layer_idx] = int(p_final)
            if state.pq_codebooks_k is not None:
                state.pq_codebooks_k[layer_idx] = cb_k
                state.pq_codebooks_v[layer_idx] = cb_v
            new_k = torch.cat([key_states[:, :, :system_size, :], new_k_vis], dim=2)
            new_v = torch.cat([value_states[:, :, :system_size, :], new_v_vis], dim=2)
            past_key_values.layers[layer_idx].keys = new_k
            past_key_values.layers[layer_idx].values = new_v
            # Bias includes system tokens (zero) + visual ProtoKV bias.
            sys_bias = torch.zeros((1, num_heads, int(system_size)), device=key_states.device, dtype=torch.float32)
            state.bias_by_layer[layer_idx] = torch.cat([sys_bias, bias_vis], dim=-1)
            continue

        # Baseline compression paths retained for compatibility.
        _, num_heads, _, _ = key_states.shape
        total_tokens_to_keep = int(compress_frame_num) * int(token_per_frame)
        if method == "swa":
            start_idx = key_states_to_compress.shape[2] - total_tokens_to_keep
            all_indices = torch.arange(start_idx, key_states_to_compress.shape[2], device=key_states.device).unsqueeze(0).expand(num_heads, -1)
        elif method == "uniform":
            total_tokens = key_states_to_compress.shape[2]
            step = total_tokens / total_tokens_to_keep
            selected = (torch.arange(total_tokens_to_keep, device=key_states.device, dtype=torch.float32) * step).to(torch.long).clamp(max=total_tokens-1)
            all_indices = selected.unsqueeze(0).expand(num_heads, -1)
        elif method == "infinipot-v":
            val_norm_score = value_states_to_compress.norm(dim=-1)[0, :, :]
            all_indices, _ = val_norm_score.topk(total_tokens_to_keep, dim=-1).indices.sort(dim=-1)
        else:
            raise ValueError(f"Unknown compression method: {method}")
        batch_indices = torch.zeros_like(all_indices)
        head_indices = torch.arange(num_heads, device=key_states.device).unsqueeze(1).expand(-1, all_indices.size(1))
        new_key_states_to_compress = key_states_to_compress[batch_indices, head_indices, all_indices].unsqueeze(0)
        new_value_states_to_compress = value_states_to_compress[batch_indices, head_indices, all_indices].unsqueeze(0)
        past_key_values.layers[layer_idx].keys = torch.cat([key_states[:, :, :system_size, :], new_key_states_to_compress], dim=2)
        past_key_values.layers[layer_idx].values = torch.cat([value_states[:, :, :system_size, :], new_value_states_to_compress], dim=2)

    if method == "prototrack-kv":
        model._protokv_bias_by_layer = state.bias_by_layer
        try:
            past_key_values.protokv_logit_bias = state.bias_by_layer
        except Exception:
            pass
    return past_key_values, None
