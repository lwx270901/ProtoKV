"""kvcache_utils_proto.py  (CUDA-event timing patch)

Changes vs. original
---------------------
*  ``process_kv_cache`` accepts a new optional ``cuda_timer`` kwarg
   (a :class:`~cuda_timing.CudaTimer` instance).
*  The ``proto_update`` region inside ``_prototrack_kv_compress_layer``
   is now measured with a ``torch.cuda.Event`` pair when ``cuda_timer``
   is provided, falling back to the original CPU dict when it is not.
*  The legacy ``timing`` dict parameter is kept for backwards compat but
   is only used when ``cuda_timer`` is None.

All other logic is identical to the original file.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Optional, Tuple

import torch

# Optional import — cuda_timing may not be present in all environments.
try:
    from cuda_timing import CudaTimer as _CudaTimer
except ImportError:
    _CudaTimer = None  # type: ignore[misc,assignment]


# ===========================================================================
# Re-export everything that was in the original file unchanged.
# Only the changed functions are redefined here; callers that import from
# this module will see the updated versions automatically.
# ===========================================================================


@dataclass
class _ProtoTrackState:
    max_proto_frames: int
    counts: list
    taus: list
    initialized: bool = False
    proto_frames_cur: int = 0
    total_frames_seen: int = 0
    pq_codebooks_k: list = None
    pq_codebooks_v: list = None
    # Residual-statistics PQ histograms.  Each layer tensor has shape
    # [num_heads, max_proto_frames, token_per_frame, S_eff, codebook_size].
    pq_hist_k: list = None
    pq_hist_v: list = None
    pq_hist_token_per_frame: int = 0
    pq_last_subspaces: int = 0
    pq_last_codebook_size: int = 0


def _get_or_reset_prototrack_state(model, num_layers, num_heads, max_proto_frames, device, reset):
    key = "_prototrack_kv_state"
    if reset or not hasattr(model, key):
        counts = [torch.zeros((num_heads, max_proto_frames), device=device, dtype=torch.float32) for _ in range(num_layers)]
        taus = [torch.zeros((num_heads, max_proto_frames), device=device, dtype=torch.long) for _ in range(num_layers)]
        state = _ProtoTrackState(max_proto_frames=max_proto_frames, counts=counts, taus=taus, initialized=False, proto_frames_cur=0, total_frames_seen=0, pq_codebooks_k=[None]*num_layers, pq_codebooks_v=[None]*num_layers, pq_hist_k=None, pq_hist_v=None, pq_hist_token_per_frame=0, pq_last_subspaces=0, pq_last_codebook_size=0)
        setattr(model, key, state)
        return state
    state = getattr(model, key)
    if not isinstance(state, _ProtoTrackState) or state.max_proto_frames != max_proto_frames:
        counts = [torch.zeros((num_heads, max_proto_frames), device=device, dtype=torch.float32) for _ in range(num_layers)]
        taus = [torch.zeros((num_heads, max_proto_frames), device=device, dtype=torch.long) for _ in range(num_layers)]
        state = _ProtoTrackState(max_proto_frames=max_proto_frames, counts=counts, taus=taus, initialized=False, proto_frames_cur=0, total_frames_seen=0, pq_codebooks_k=[None]*num_layers, pq_codebooks_v=[None]*num_layers, pq_hist_k=None, pq_hist_v=None, pq_hist_token_per_frame=0, pq_last_subspaces=0, pq_last_codebook_size=0)
        setattr(model, key, state)
    if len(state.counts) != num_layers:
        state.counts = [torch.zeros((num_heads, max_proto_frames), device=device, dtype=torch.float32) for _ in range(num_layers)]
        state.taus = [torch.zeros((num_heads, max_proto_frames), device=device, dtype=torch.long) for _ in range(num_layers)]
        state.initialized = False
        state.proto_frames_cur = 0
        state.total_frames_seen = 0
        state.pq_codebooks_k = [None]*num_layers
        state.pq_codebooks_v = [None]*num_layers
        state.pq_hist_k = None
        state.pq_hist_v = None
        state.pq_hist_token_per_frame = 0
        state.pq_last_subspaces = 0
        state.pq_last_codebook_size = 0
    return state


_INV_FREQ_CACHE: dict = {}


def _infer_rope_theta(model, default: float = 10000.0) -> float:
    try:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            for name in ("rope_theta", "rotary_emb_base", "rotary_base", "rope_base"):
                if hasattr(cfg, name):
                    v = getattr(cfg, name)
                    if v is not None:
                        return float(v)
    except Exception:
        pass
    return float(default)


def _rope_inv_freq(head_dim: int, rope_theta: float, device) -> torch.Tensor:
    key = (int(head_dim), float(rope_theta), device)
    inv = _INV_FREQ_CACHE.get(key)
    if inv is None or inv.device != device:
        d = int(head_dim)
        inv = 1.0 / (float(rope_theta) ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
        _INV_FREQ_CACHE[key] = inv
    return inv


def _apply_rope_delta(x: torch.Tensor, delta: torch.Tensor, rope_theta: float) -> torch.Tensor:
    if x.numel() == 0 or torch.all(delta == 0):
        return x
    H, TPF, D = x.shape
    inv_freq = _rope_inv_freq(D, rope_theta, x.device)
    ang = delta.to(torch.float32).view(H, 1) * inv_freq.view(1, -1)
    cos = torch.cos(ang)[:, None, :]
    sin = torch.sin(ang)[:, None, :]
    x_f = x.to(torch.float32)
    out_even = x_f[..., 0::2] * cos - x_f[..., 1::2] * sin
    out_odd = x_f[..., 0::2] * sin + x_f[..., 1::2] * cos
    out = torch.empty_like(x_f)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out.to(dtype=x.dtype)


def _frame_embed_from_keys(keys_frame: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    emb = keys_frame.mean(dim=1).float()
    return torch.nn.functional.normalize(emb, dim=-1, eps=eps)


def _proto_embed_from_keys(proto_k: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    emb = proto_k.mean(dim=2).float()
    return torch.nn.functional.normalize(emb, dim=-1, eps=eps)


def _nearest_divisor(n: int, target: int) -> int:
    target = int(max(1, target))
    if n % target == 0:
        return target
    divs = [d for d in range(1, n + 1) if n % d == 0]
    divs.sort(key=lambda d: (abs(d - target), -d))
    return divs[0]


@torch.no_grad()
def _pq_fit_codebooks(x, subspaces, codebook_size, iters=4, sample_size=4096, seed=0):
    N, D = x.shape
    S, M, ds = int(subspaces), int(codebook_size), D // int(subspaces)
    if N > sample_size:
        g = torch.Generator(device=x.device); g.manual_seed(int(seed))
        x_fit = x[torch.randperm(N, device=x.device, generator=g)[:sample_size]]
    else:
        x_fit = x
    x_fit = x_fit.float()
    codebooks = torch.empty((S, M, ds), device=x.device, dtype=torch.float32)
    for s in range(S):
        xs = x_fit[:, s*ds:(s+1)*ds]
        g = torch.Generator(device=x.device); g.manual_seed(int(seed + 17*s))
        centroids = xs[torch.randperm(xs.shape[0], device=x.device, generator=g)[:M]].clone()
        for _ in range(int(iters)):
            dist = (xs**2).sum(1, keepdim=True) + (centroids**2).sum(1).view(1,M) - 2*(xs @ centroids.t())
            assign = dist.argmin(1)
            sums = torch.zeros_like(centroids); counts = torch.zeros((M,), device=x.device, dtype=torch.float32)
            sums.index_add_(0, assign, xs); ones = torch.ones((xs.shape[0],), device=x.device, dtype=torch.float32); counts.index_add_(0, assign, ones)
            empty = counts <= 0.0
            if empty.any():
                centroids[empty] = xs[torch.randperm(xs.shape[0], device=x.device, generator=g)[:int(empty.sum().item())]]
                counts[empty] = 1.0
            centroids = sums / counts.clamp(min=1.0).unsqueeze(1)
        codebooks[s] = centroids
    return codebooks.to(torch.float16)


@torch.no_grad()
def _pq_encode_decode(x, codebooks):
    N, D = x.shape; S, M, ds = codebooks.shape
    x = x.float(); cb = codebooks.float(); parts = []
    for s in range(S):
        xs = x[:, s*ds:(s+1)*ds]; cs = cb[s]
        dist = (xs**2).sum(1,keepdim=True) + (cs**2).sum(1).view(1,M) - 2*(xs @ cs.t())
        parts.append(cs[dist.argmin(1)])
    return torch.cat(parts, dim=1)



@torch.no_grad()
def _pq_encode_indices(x: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """Encode vectors with PQ codebooks and return integer code indices.

    Parameters
    ----------
    x : Tensor [N, D]
        Residual vectors.
    codebooks : Tensor [S, M, d_s]
        PQ codebooks.

    Returns
    -------
    Tensor [N, S] of int64 codeword indices.
    """
    N, D = x.shape
    S, M, ds = codebooks.shape
    if D != S * ds:
        raise ValueError(f"PQ dimension mismatch: x D={D}, codebooks S*ds={S*ds}")
    x_f = x.float()
    cb = codebooks.float()
    codes = torch.empty((N, S), device=x.device, dtype=torch.long)
    for s in range(S):
        xs = x_f[:, s * ds:(s + 1) * ds]
        cs = cb[s]
        dist = (xs ** 2).sum(1, keepdim=True) + (cs ** 2).sum(1).view(1, M) - 2 * (xs @ cs.t())
        codes[:, s] = dist.argmin(1)
    return codes


def _zero_pq_histograms_for_slots(hist_k: Optional[torch.Tensor], hist_v: Optional[torch.Tensor], start: int, end: int) -> None:
    if hist_k is not None:
        hist_k[:, start:end].zero_()
    if hist_v is not None:
        hist_v[:, start:end].zero_()


@torch.no_grad()
def _update_pq_residual_histograms(
    hist_k: Optional[torch.Tensor],
    hist_v: Optional[torch.Tensor],
    frame_k: torch.Tensor,
    frame_v: torch.Tensor,
    proto_k: torch.Tensor,
    proto_v: torch.Tensor,
    assign_idx: torch.Tensor,
    codebook_k: Optional[torch.Tensor],
    codebook_v: Optional[torch.Tensor],
    frame_tau: Optional[torch.Tensor] = None,
    proto_taus: Optional[torch.Tensor] = None,
    rope_theta: float = 10000.0,
) -> None:
    """Update marginal PQ histograms H_k^K and H_k^V for one absorbed frame.

    hist_* shape: [H, Pmax, TPF, S, M].
    frame_* shape: [H, TPF, D].
    proto_* shape: [H, Pcur, TPF, D].
    assign_idx shape: [H].

    The update implements:
        r^K = K_e - c_k, r^V = V_e - p_k,
        z_g = nearest PQ codeword, H[g,z_g] += 1.
    """
    if hist_k is None or hist_v is None or codebook_k is None or codebook_v is None:
        return
    H, TPF, D = frame_k.shape
    device = frame_k.device
    h_idx = torch.arange(H, device=device)
    fk = frame_k
    if frame_tau is not None and proto_taus is not None:
        target_tau = proto_taus[h_idx, assign_idx].to(torch.long)
        ft = frame_tau.view(1).expand(H).to(torch.long) if frame_tau.numel() == 1 else frame_tau.to(torch.long).view(-1)
        fk = _apply_rope_delta(frame_k, (target_tau - ft).to(torch.float32), rope_theta=rope_theta)
    pk = proto_k[h_idx, assign_idx]
    pv = proto_v[h_idx, assign_idx]
    rK = (fk.float() - pk.float()).reshape(H * TPF, D)
    rV = (frame_v.float() - pv.float()).reshape(H * TPF, D)
    codesK = _pq_encode_indices(rK, codebook_k).reshape(H, TPF, -1)
    codesV = _pq_encode_indices(rV, codebook_v).reshape(H, TPF, -1)
    S_eff = codesK.shape[-1]
    token_idx = torch.arange(TPF, device=device)
    for h in range(H):
        kk = int(assign_idx[h].item())
        for s in range(S_eff):
            hist_k[h, kk, token_idx, s, codesK[h, :, s]] += 1
            hist_v[h, kk, token_idx, s, codesV[h, :, s]] += 1


@torch.no_grad()
def _merge_pq_histograms_by_assignment(
    hist_k: Optional[torch.Tensor],
    hist_v: Optional[torch.Tensor],
    src_slot: int,
    assign_idx: torch.Tensor,
) -> None:
    """Merge an obsolete prototype slot's residual histograms into assigned kept slots."""
    if hist_k is None or hist_v is None:
        return
    H = hist_k.shape[0]
    for h in range(H):
        dst = int(assign_idx[h].item())
        hist_k[h, dst] += hist_k[h, src_slot]
        hist_v[h, dst] += hist_v[h, src_slot]
        hist_k[h, src_slot].zero_()
        hist_v[h, src_slot].zero_()


@torch.no_grad()
def _pq_expected_residual_from_hist(hist: torch.Tensor, codebooks: torch.Tensor) -> torch.Tensor:
    """Decode the expected PQ residual from marginal histograms.

    hist:      [H, P, TPF, S, M]
    codebooks: [S, M, ds]
    returns:   [H, P, TPF, D]

    This is the S=1 readout from the histogram distribution.  It preserves
    the fixed output length used by the existing inference code while making
    prototype output depend on accumulated residual statistics.
    """
    if hist.numel() == 0:
        return torch.empty((*hist.shape[:3], 0), device=hist.device)
    denom = hist.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
    probs = hist.float() / denom
    # [H,P,TPF,S,M] x [S,M,ds] -> [H,P,TPF,S,ds]
    parts = torch.einsum("hptsm,smd->hptsd", probs, codebooks.float())
    H, P, TPF, S_eff, ds = parts.shape
    return parts.reshape(H, P, TPF, S_eff * ds)




@torch.no_grad()
def _pq_decode_top_s_residuals_from_hist(
    hist: torch.Tensor,
    codebooks: torch.Tensor,
    top_s: int = 1,
    beam_size: Optional[int] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Decode top-S full PQ residual modes using Algorithm 3 beam search.

    hist:      [H, P, TPF, G, C]
    codebooks: [G, C, d_g]
    return:    [H, P, S, TPF, D]

    H[g,c] is treated as a factorized categorical distribution over PQ
    subquantizers. We keep the top-B partial tuples at each subquantizer and
    return the top-S full codeword tuples. Empty histograms return zero residuals.
    """
    if hist.dim() != 5 or codebooks.dim() != 3:
        raise ValueError(f"Expected hist [H,P,TPF,G,C] and codebooks [G,C,dg], got {tuple(hist.shape)} and {tuple(codebooks.shape)}")
    H, P, TPF, G, C = hist.shape
    Gc, Cc, ds = codebooks.shape
    if G != Gc or C != Cc:
        raise ValueError(f"PQ hist/codebook mismatch: hist G,C={(G, C)}, codebook G,C={(Gc, Cc)}")

    S_out = max(1, int(top_s))
    B = 4 * S_out if beam_size is None or int(beam_size) <= 0 else max(S_out, int(beam_size))

    N = H * P * TPF
    flat_hist = hist.reshape(N, G, C).float()
    valid = flat_hist.sum(dim=(1, 2)) > 0
    probs = (flat_hist + float(eps)) / (flat_hist.sum(dim=-1, keepdim=True) + float(C) * float(eps)).clamp(min=1e-12)
    logp = torch.log(probs)

    beam_scores = torch.zeros((N, 1), device=hist.device, dtype=torch.float32)
    beam_codes = torch.empty((N, 1, 0), device=hist.device, dtype=torch.long)
    for g in range(G):
        cand = beam_scores.unsqueeze(-1) + logp[:, g, :].unsqueeze(1)  # [N,Bcur,C]
        flat = cand.reshape(N, -1)
        keep = min(B, flat.shape[1])
        beam_scores, top_idx = torch.topk(flat, k=keep, dim=1)
        prev_beam = top_idx // C
        new_code = top_idx % C
        if g == 0:
            beam_codes = new_code.unsqueeze(-1)
        else:
            prev_codes = beam_codes.gather(1, prev_beam.unsqueeze(-1).expand(-1, -1, g))
            beam_codes = torch.cat([prev_codes, new_code.unsqueeze(-1)], dim=-1)

    take = min(S_out, beam_codes.shape[1])
    codes = beam_codes[:, :take, :]
    if take < S_out:
        codes = torch.cat([codes, codes[:, -1:, :].expand(-1, S_out - take, -1)], dim=1)

    cb = codebooks.float()
    parts = [cb[g][codes[:, :, g]] for g in range(G)]
    residual = torch.cat(parts, dim=-1)  # [N,S,D]
    residual = residual * valid.view(N, 1, 1).to(residual.dtype)
    return residual.reshape(H, P, TPF, S_out, G * ds).permute(0, 1, 3, 2, 4).contiguous()

@torch.no_grad()
def _fit_residual_codebooks_from_frames(
    far_k: torch.Tensor,
    far_v: torch.Tensor,
    proto_k: torch.Tensor,
    proto_v: torch.Tensor,
    subspaces: int,
    codebook_size: int,
    iters: int,
    sample_size: int,
    seed: int,
    rope_theta: float = 10000.0,
    far_frame_taus: Optional[torch.Tensor] = None,
    proto_taus: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fit shared per-layer/head PQ codebooks on residuals around current prototypes."""
    H, F, TPF, D = far_k.shape
    if F <= 0:
        raise ValueError("Cannot fit residual PQ codebooks without far frames.")
    proto_emb = _proto_embed_from_keys(proto_k)
    assign = []
    for f in range(F):
        assign.append(_assign_frames_to_prototypes(far_k[:, f], proto_emb=proto_emb))
    assign = torch.stack(assign, dim=1)  # [H,F]
    h_idx = torch.arange(H, device=far_k.device).view(H, 1).expand(H, F)
    pk = proto_k[h_idx, assign]  # [H,F,TPF,D]
    pv = proto_v[h_idx, assign]
    fk = far_k
    if far_frame_taus is not None and proto_taus is not None:
        # Align each frame key to the assigned prototype timestamp before residual fitting.
        target_tau = proto_taus[h_idx, assign].to(torch.long)  # [H,F]
        ft = far_frame_taus.to(torch.long).view(1, F).expand(H, F)
        # _apply_rope_delta works on [H,TPF,D], so loop over F for clarity.
        aligned = []
        for f in range(F):
            aligned.append(_apply_rope_delta(far_k[:, f], (target_tau[:, f] - ft[:, f]).to(torch.float32), rope_theta=rope_theta))
        fk = torch.stack(aligned, dim=1)
    rK = (fk.float() - pk.float()).reshape(H * F * TPF, D)
    rV = (far_v.float() - pv.float()).reshape(H * F * TPF, D)
    S_eff = _nearest_divisor(D, int(subspaces))
    cb_k = _pq_fit_codebooks(rK, S_eff, int(codebook_size), iters, sample_size, seed=seed + 11)
    cb_v = _pq_fit_codebooks(rV, S_eff, int(codebook_size), iters, sample_size, seed=seed + 29)
    return cb_k, cb_v


@torch.no_grad()
def _rebuild_pq_histograms_from_frames(
    hist_k: torch.Tensor,
    hist_v: torch.Tensor,
    far_k: torch.Tensor,
    far_v: torch.Tensor,
    proto_k: torch.Tensor,
    proto_v: torch.Tensor,
    codebook_k: torch.Tensor,
    codebook_v: torch.Tensor,
    rope_theta: float = 10000.0,
    far_frame_taus: Optional[torch.Tensor] = None,
    proto_taus: Optional[torch.Tensor] = None,
) -> None:
    """Clear and rebuild residual histograms from available far frames."""
    hist_k.zero_(); hist_v.zero_()
    H, F, _, _ = far_k.shape
    proto_emb = _proto_embed_from_keys(proto_k)
    for f in range(F):
        assign = _assign_frames_to_prototypes(far_k[:, f], proto_emb=proto_emb)
        ft = far_frame_taus[f].view(1) if far_frame_taus is not None else None
        _update_pq_residual_histograms(
            hist_k, hist_v,
            far_k[:, f], far_v[:, f],
            proto_k, proto_v,
            assign,
            codebook_k, codebook_v,
            frame_tau=ft,
            proto_taus=proto_taus,
            rope_theta=rope_theta,
        )


@torch.no_grad()
def _pq_quantize_prototypes(proto, subspaces, codebook_size, iters, sample_size, codebook, seed=0):
    D = proto.shape[-1]; S_eff = _nearest_divisor(D, subspaces); M = int(codebook_size); ds = D // S_eff
    x = proto.reshape(-1, D)
    if codebook is None or codebook.dim() != 3 or codebook.shape != (S_eff, M, ds):
        codebook = _pq_fit_codebooks(x, S_eff, M, iters, sample_size, seed)
    return _pq_encode_decode(x, codebook).to(proto.dtype).reshape_as(proto), codebook


def _assign_frames_to_prototypes(frame_k, proto_k=None, proto_emb=None, eps=1e-6):
    f = _frame_embed_from_keys(frame_k, eps=eps)
    if proto_emb is None:
        if proto_k is None:
            raise ValueError("Either proto_k or proto_emb must be provided.")
        p = _proto_embed_from_keys(proto_k, eps=eps)
    else:
        p = proto_emb
    return (p * f.unsqueeze(1)).sum(dim=-1).argmax(dim=-1)


def _refresh_proto_embed_inplace(
    proto_k: torch.Tensor,
    proto_emb: torch.Tensor,
    assign_idx: torch.Tensor,
    h_idx: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    updated = proto_k[h_idx, assign_idx]
    emb = updated.mean(dim=1).float()
    proto_emb[h_idx, assign_idx] = torch.nn.functional.normalize(emb, dim=-1, eps=eps)


def _running_mean_update(proto, frame, counts, assign_idx, weight=None, h_idx: Optional[torch.Tensor] = None):
    device = proto.device; H = proto.shape[0]
    if h_idx is None:
        h_idx = torch.arange(H, device=device)
    cur = proto[h_idx, assign_idx]; c = counts[h_idx, assign_idx]
    w = torch.ones_like(c) if weight is None else weight.to(c.dtype)
    lr = (w / (c + w)).view(H, 1, 1).to(cur.dtype)
    proto[h_idx, assign_idx] = (cur.float() + (frame.float() - cur.float()) * lr.float()).to(cur.dtype)
    counts[h_idx, assign_idx] = c + w


def _running_mean_update_keys_rope_tau_last(
    proto_k, frame_k, counts, taus, assign_idx, frame_tau, rope_theta, weight=None, h_idx: Optional[torch.Tensor] = None
):
    device = proto_k.device; H = proto_k.shape[0]
    if h_idx is None:
        h_idx = torch.arange(H, device=device)
    cur = proto_k[h_idx, assign_idx]; tau_old = taus[h_idx, assign_idx].to(torch.long)
    tau_frame = frame_tau.view(1).expand(H).to(torch.long) if frame_tau.dim() == 0 else frame_tau.to(torch.long).view(-1).expand(H) if frame_tau.numel() == 1 else frame_tau.to(torch.long).view(-1)
    frame_aligned = _apply_rope_delta(frame_k, (tau_old - tau_frame).to(torch.float32), rope_theta=rope_theta)
    c = counts[h_idx, assign_idx]
    w = torch.ones_like(c) if weight is None else weight.to(c.dtype)
    lr = (w / (c + w)).view(H, 1, 1).to(cur.dtype)
    new = (cur.float() + (frame_aligned.float() - cur.float()) * lr.float()).to(cur.dtype)
    counts[h_idx, assign_idx] = c + w
    tau_new = torch.maximum(tau_old, tau_frame)
    proto_k[h_idx, assign_idx] = _apply_rope_delta(new, (tau_new - tau_old).to(torch.float32), rope_theta=rope_theta)
    taus[h_idx, assign_idx] = tau_new


def _init_prototypes_from_far_frames(
    far_k,
    far_v,
    proto_frames,
    counts,
    taus=None,
    far_frame_taus=None,
    rope_theta=10000.0,
    h_idx: Optional[torch.Tensor] = None,
    proto_update_ctx_factory=None,
):
    H, F, TPF, D = far_k.shape; device = far_k.device
    if h_idx is None:
        h_idx = torch.arange(H, device=device)
    if F <= proto_frames:
        seed_idx = torch.arange(F, device=device)
        if F < proto_frames:
            seed_idx = torch.cat([seed_idx, torch.full((proto_frames-F,), int(F-1), device=device, dtype=torch.long)])
    else:
        seed_idx = torch.round(torch.linspace(0, F-1, steps=proto_frames, device=device)).long()
    proto_k = far_k[:, seed_idx].contiguous(); proto_v = far_v[:, seed_idx].contiguous()
    proto_emb = _proto_embed_from_keys(proto_k)
    counts[:, :proto_frames].zero_(); counts[:, :proto_frames] += 1.0
    if taus is not None:
        if far_frame_taus is None:
            far_frame_taus = torch.arange(F, device=device, dtype=torch.long)
        taus[:, :proto_frames].zero_()
        seed_taus = far_frame_taus[seed_idx].to(torch.long) if far_frame_taus is not None else seed_idx.to(torch.long)
        taus[:, :proto_frames] = seed_taus.view(1, -1).expand(H, -1)
    if F > 0:
        seed_mask = torch.zeros((F,), device=device, dtype=torch.bool)
        seed_mask[seed_idx.clamp(0, F - 1)] = True
        update_idx = torch.nonzero(~seed_mask, as_tuple=False).flatten()
    else:
        update_idx = seed_idx.new_empty((0,))

    for f in update_idx:
        fk = far_k[:, f]; fv = far_v[:, f]
        assign = _assign_frames_to_prototypes(fk, proto_emb=proto_emb)
        if proto_update_ctx_factory is not None:
            _ctx = proto_update_ctx_factory()
        else:
            _ctx = _NullCtx(timing=None)
        with _ctx:
            if taus is not None:
                tau_f = far_frame_taus[f]
                _running_mean_update_keys_rope_tau_last(
                    proto_k, fk, counts[:, :proto_frames], taus[:, :proto_frames], assign, tau_f, rope_theta=rope_theta, h_idx=h_idx
                )
            else:
                _running_mean_update(proto_k, fk, counts[:, :proto_frames], assign, h_idx=h_idx)
        _refresh_proto_embed_inplace(proto_k, proto_emb, assign, h_idx)
        _running_mean_update(proto_v, fv, counts[:, :proto_frames], assign, h_idx=h_idx)
    return proto_k, proto_v


# ---------------------------------------------------------------------------
# Modified _prototrack_kv_compress_layer
# The only change: the proto_update timing block now accepts an optional
# cuda_timer and records with CUDA events when available.
# ---------------------------------------------------------------------------

def _prototrack_kv_compress_layer(
    key_states_to_compress: torch.Tensor,
    value_states_to_compress: torch.Tensor,
    token_per_frame: int,
    compress_frame_num: int,
    proto_frames_max: int,
    state_counts: torch.Tensor,
    state_taus: Optional[torch.Tensor],
    state_initialized: bool,
    state_proto_frames_cur: int,
    cur_frame_end: Optional[int] = None,
    rope_theta: float = 10000.0,
    pq_subspaces: int = 0,
    pq_codebook_size: int = 16,
    pq_kmeans_iters: int = 4,
    pq_sample_size: int = 4096,
    pq_codebook_k: Optional[torch.Tensor] = None,
    pq_codebook_v: Optional[torch.Tensor] = None,
    pq_hist_k: Optional[torch.Tensor] = None,
    pq_hist_v: Optional[torch.Tensor] = None,
    pq_seed: int = 0,
    pq_decode_top_s: int = 1,
    pq_decode_beam_size: int = 0,
    pq_decode_eps: float = 1e-5,
    # ── NEW: prefer cuda_timer over the legacy timing dict ──────────────
    cuda_timer=None,          # Optional[CudaTimer]
    timing: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, bool, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """ProtoTrack-KV compression for a single layer.

    The proto-update region is now measured with CUDA events via
    ``cuda_timer("proto_update")`` when ``cuda_timer`` is provided.
    """
    assert key_states_to_compress.dim() == 4 and key_states_to_compress.shape[0] == 1
    _, H, L, D = key_states_to_compress.shape
    assert L % token_per_frame == 0
    cur_frames = L // token_per_frame
    use_rope_align = (state_taus is not None) and (cur_frame_end is not None)

    if cur_frames <= compress_frame_num:
        return key_states_to_compress, value_states_to_compress, state_initialized, int(state_proto_frames_cur), pq_codebook_k, pq_codebook_v

    # Top-S decoding turns each prototype slot into S pseudo-token frames.
    # Keep the final KV length fixed by reducing prototype slots accordingly.
    decode_top_s_eff = max(1, int(pq_decode_top_s))
    if compress_frame_num <= decode_top_s_eff:
        decode_top_s_eff = 1
    proto_frames_target = min(proto_frames_max, max(1, (compress_frame_num - 1) // decode_top_s_eff))
    recent_frames = compress_frame_num - proto_frames_target * decode_top_s_eff
    assert recent_frames >= 1

    kf = key_states_to_compress[0].reshape(H, cur_frames, token_per_frame, D)
    vf = value_states_to_compress[0].reshape(H, cur_frames, token_per_frame, D)
    device = kf.device
    h_idx = torch.arange(H, device=device)
    cur_frame_end_i = int(cur_frame_end) if cur_frame_end is not None else 0

    def _tau_start(P_proto: int) -> int:
        # Equivalent to old _exact_frame_tau(global_idx, P_proto) evaluated at global_idx == P_proto.
        return cur_frame_end_i - max(0, cur_frames - P_proto)

    near_start = cur_frames - recent_frames
    near_k = kf[:, near_start:]
    near_v = vf[:, near_start:]

    # ``proto_update`` is intentionally strict: only prototype-state update math.
    # Assignment/search/index bookkeeping are excluded from this timer.
    if cuda_timer is not None:
        def _proto_update_ctx_factory():
            return cuda_timer("proto_update")
    else:
        def _proto_update_ctx_factory():
            return _NullCtx(timing=timing)

    if not state_initialized or state_proto_frames_cur <= 0:
        far_frames = near_start
        far_k = kf[:, :far_frames]; far_v = vf[:, :far_frames]
        if use_rope_align:
            far_taus = torch.arange(far_frames, device=device, dtype=torch.long)
            proto_k, proto_v = _init_prototypes_from_far_frames(
                far_k,
                far_v,
                proto_frames_target,
                state_counts,
                taus=state_taus,
                far_frame_taus=far_taus,
                rope_theta=rope_theta,
                h_idx=h_idx,
                proto_update_ctx_factory=_proto_update_ctx_factory,
            )
        else:
            proto_k, proto_v = _init_prototypes_from_far_frames(
                far_k,
                far_v,
                proto_frames_target,
                state_counts,
                h_idx=h_idx,
                proto_update_ctx_factory=_proto_update_ctx_factory,
            )
        state_initialized = True; state_proto_frames_cur = proto_frames_target
    else:
        P_old = min(int(state_proto_frames_cur), cur_frames); P_old = max(1, P_old)
        proto_k = kf[:, :P_old].contiguous(); proto_v = vf[:, :P_old].contiguous()
        proto_emb = _proto_embed_from_keys(proto_k)
        if float(state_counts[:, :P_old].sum().item()) == 0.0:
            state_counts[:, :P_old] += 1.0

        P_new = proto_frames_target

        if P_new < P_old:
            extra_k = proto_k[:, P_new:P_old]; extra_v = proto_v[:, P_new:P_old]
            extra_counts = state_counts[:, P_new:P_old].clone()
            extra_taus = state_taus[:, P_new:P_old].clone() if use_rope_align else None
            proto_k = proto_k[:, :P_new].contiguous(); proto_v = proto_v[:, :P_new].contiguous()
            proto_emb = _proto_embed_from_keys(proto_k)
            counts_kept = state_counts[:, :P_new]; taus_kept = state_taus[:, :P_new] if use_rope_align else None
            for e in range(extra_k.shape[1]):
                fk = extra_k[:, e]; fv = extra_v[:, e]; w = extra_counts[:, e]
                tau_e = extra_taus[:, e] if use_rope_align else None
                assign = _assign_frames_to_prototypes(fk, proto_emb=proto_emb)
                with _proto_update_ctx_factory():
                    if use_rope_align:
                        _running_mean_update_keys_rope_tau_last(
                            proto_k, fk, counts_kept, taus_kept, assign, tau_e, rope_theta=rope_theta, weight=w, h_idx=h_idx
                        )
                    else:
                        _running_mean_update(proto_k, fk, counts_kept, assign, weight=w, h_idx=h_idx)
                _refresh_proto_embed_inplace(proto_k, proto_emb, assign, h_idx)
                _running_mean_update(proto_v, fv, counts_kept, assign, weight=w, h_idx=h_idx)
                # Merge residual histograms from removed prototype slots into
                # their assigned surviving prototypes.  This preserves H_k
                # without needing the original per-token residuals.
                _merge_pq_histograms_by_assignment(pq_hist_k, pq_hist_v, P_new + e, assign)
            P_old = P_new; state_proto_frames_cur = P_new

        if P_new > P_old:
            need = P_new - P_old; take = min(need, max(0, near_start - P_old))
            if take > 0:
                seed_start = P_old
                seed_idx = torch.arange(P_old, P_old + take, device=device)
                proto_k = torch.cat([proto_k, kf[:, seed_idx].contiguous()], dim=1)
                proto_v = torch.cat([proto_v, vf[:, seed_idx].contiguous()], dim=1)
                state_counts[:, P_old:P_old+take].zero_(); state_counts[:, P_old:P_old+take] += 1.0
                _zero_pq_histograms_for_slots(pq_hist_k, pq_hist_v, P_old, P_old + take)
                if use_rope_align:
                    tau_start_seed = _tau_start(seed_start)
                    seed_taus = (seed_idx - seed_start + tau_start_seed).to(torch.long)
                    state_taus[:, P_old:P_old+take] = seed_taus.view(1, -1).expand(H, -1)
                P_old = P_old + take; state_proto_frames_cur = P_old
                proto_emb = _proto_embed_from_keys(proto_k)

        P_cur = P_old
        if near_start > P_cur:
            exact_k = kf[:, P_cur:near_start]; exact_v = vf[:, P_cur:near_start]
            counts_cur = state_counts[:, :P_cur]
            taus_cur = state_taus[:, :P_cur] if use_rope_align else None
            tau_seq = None
            if use_rope_align:
                tau_seq = torch.arange(exact_k.shape[1], device=device, dtype=torch.long) + _tau_start(P_cur)
            for f in range(exact_k.shape[1]):
                fk = exact_k[:, f]; fv = exact_v[:, f]
                assign = _assign_frames_to_prototypes(fk, proto_emb=proto_emb)
                with _proto_update_ctx_factory():
                    if use_rope_align:
                        tau_f = tau_seq[f]
                        _running_mean_update_keys_rope_tau_last(
                            proto_k, fk, counts_cur, taus_cur, assign, tau_f, rope_theta=rope_theta, h_idx=h_idx
                        )
                    else:
                        _running_mean_update(proto_k, fk, counts_cur, assign, h_idx=h_idx)
                _refresh_proto_embed_inplace(proto_k, proto_emb, assign, h_idx)
                _running_mean_update(proto_v, fv, counts_cur, assign, h_idx=h_idx)
                if pq_subspaces and int(pq_subspaces) > 0 and pq_codebook_k is not None and pq_codebook_v is not None:
                    ft = tau_seq[f].view(1) if use_rope_align else None
                    _update_pq_residual_histograms(
                        pq_hist_k, pq_hist_v,
                        fk, fv,
                        proto_k[:, :P_cur], proto_v[:, :P_cur],
                        assign,
                        pq_codebook_k, pq_codebook_v,
                        frame_tau=ft,
                        proto_taus=taus_cur,
                        rope_theta=rope_theta,
                    )

    # ────────────────────────────────────────────────────────────────────

    P_final = min(proto_frames_target, proto_k.shape[1])
    recent_final = compress_frame_num - P_final * decode_top_s_eff
    near_k_final = kf[:, cur_frames - recent_final:]
    near_v_final = vf[:, cur_frames - recent_final:]

    # Algorithm 3 + Algorithm 4 path: decode top-S residual modes and expose
    # S pseudo-token frames per prototype. Key and Value modes are decoded
    # separately from H^K/H^V and paired by rank s.
    proto_center_k = proto_k[:, :P_final].contiguous()
    proto_center_v = proto_v[:, :P_final].contiguous()
    proto_modes_k = proto_center_k.unsqueeze(2).expand(-1, -1, decode_top_s_eff, -1, -1).contiguous()
    proto_modes_v = proto_center_v.unsqueeze(2).expand(-1, -1, decode_top_s_eff, -1, -1).contiguous()

    if pq_subspaces and int(pq_subspaces) > 0 and int(pq_codebook_size) > 1 and P_final > 0 and pq_hist_k is not None and pq_hist_v is not None:
        far_frames_for_hist = max(0, near_start)
        far_k_hist = kf[:, :far_frames_for_hist]
        far_v_hist = vf[:, :far_frames_for_hist]
        far_taus_hist = None
        if use_rope_align:
            far_taus_hist = torch.arange(far_frames_for_hist, device=device, dtype=torch.long)
        if (pq_codebook_k is None or pq_codebook_v is None) and far_frames_for_hist > 0:
            pq_codebook_k, pq_codebook_v = _fit_residual_codebooks_from_frames(
                far_k_hist, far_v_hist,
                proto_center_k, proto_center_v,
                int(pq_subspaces), int(pq_codebook_size), int(pq_kmeans_iters), int(pq_sample_size), int(pq_seed),
                rope_theta=rope_theta,
                far_frame_taus=far_taus_hist,
                proto_taus=(state_taus[:, :P_final] if use_rope_align else None),
            )
        if pq_codebook_k is not None and pq_codebook_v is not None and far_frames_for_hist > 0 and int(pq_hist_k[:, :P_final].sum().item()) == 0:
            _rebuild_pq_histograms_from_frames(
                pq_hist_k[:, :P_final], pq_hist_v[:, :P_final],
                far_k_hist, far_v_hist,
                proto_center_k, proto_center_v,
                pq_codebook_k, pq_codebook_v,
                rope_theta=rope_theta,
                far_frame_taus=far_taus_hist,
                proto_taus=(state_taus[:, :P_final] if use_rope_align else None),
            )
        if pq_codebook_k is not None and pq_codebook_v is not None:
            rK_modes = _pq_decode_top_s_residuals_from_hist(
                pq_hist_k[:, :P_final], pq_codebook_k,
                top_s=decode_top_s_eff,
                beam_size=int(pq_decode_beam_size),
                eps=float(pq_decode_eps),
            ).to(proto_center_k.dtype)
            rV_modes = _pq_decode_top_s_residuals_from_hist(
                pq_hist_v[:, :P_final], pq_codebook_v,
                top_s=decode_top_s_eff,
                beam_size=int(pq_decode_beam_size),
                eps=float(pq_decode_eps),
            ).to(proto_center_v.dtype)
            if rK_modes.shape == proto_modes_k.shape:
                proto_modes_k = proto_center_k.unsqueeze(2) + rK_modes
            if rV_modes.shape == proto_modes_v.shape:
                proto_modes_v = proto_center_v.unsqueeze(2) + rV_modes

    proto_out_k = proto_modes_k.reshape(H, P_final * decode_top_s_eff, token_per_frame, D)
    proto_out_v = proto_modes_v.reshape(H, P_final * decode_top_s_eff, token_per_frame, D)
    new_kf = torch.cat([proto_out_k, near_k_final], dim=1)
    new_vf = torch.cat([proto_out_v, near_v_final], dim=1)
    assert new_kf.shape[1] == compress_frame_num, (new_kf.shape[1], compress_frame_num, P_final, decode_top_s_eff, recent_final)
    new_k = new_kf.reshape(1, H, compress_frame_num * token_per_frame, D)
    new_v = new_vf.reshape(1, H, compress_frame_num * token_per_frame, D)
    return new_k, new_v, state_initialized, P_final, pq_codebook_k, pq_codebook_v


# ---------------------------------------------------------------------------
# Null-context shim: CPU fallback for proto_update timing
# when cuda_timer is not available.
# ---------------------------------------------------------------------------

class _NullCtx:
    """CPU-fallback context manager that mimics CudaTimer for proto_update."""
    def __init__(self, timing: Optional[Dict[str, float]] = None) -> None:
        self._timing = timing
        self._t0: float = 0.0

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *_):
        if self._timing is not None:
            elapsed = time.time() - self._t0
            self._timing["proto_update_s"] = self._timing.get("proto_update_s", 0.0) + elapsed


# ---------------------------------------------------------------------------
# Modified process_kv_cache — accepts cuda_timer kwarg
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
    prototrack_proto_frames: int = 2,
    prototrack_frame_end: Optional[int] = None,
    prototrack_pq_subspaces: int = 0,
    prototrack_pq_codebook_size: int = 16,
    prototrack_pq_kmeans_iters: int = 4,
    prototrack_pq_sample_size: int = 4096,
    prototrack_pq_seed: int = 0,
    prototrack_decode_top_s: int = 1,
    prototrack_decode_beam_size: int = 0,
    prototrack_decode_eps: float = 1e-5,
    # ── NEW: CUDA event timer (preferred over legacy timing dict) ────────
    cuda_timer=None,          # Optional[CudaTimer]
    timing: Optional[Dict[str, float]] = None,
) -> Tuple:
    """Compress the KV cache.

    Parameters
    ----------
    cuda_timer : CudaTimer, optional
        When provided, ``proto_update`` is measured with CUDA events via
        ``cuda_timer("proto_update")``.  This is more accurate than the CPU
        wall-clock measurement used by the legacy ``timing`` dict.
    timing : dict, optional
        Legacy CPU-timing dict.  Only used when ``cuda_timer`` is None.
    """
    pooling_func_list = [3, 2, 1, -1]
    pool_size_list = [7, 5, 3, 1]

    if compress_frame_num <= 0:
        return past_key_values, None

    current_seq_len = past_key_values.get_seq_length()
    vision_start = system_size
    vision_end = current_seq_len
    vision_length = vision_end - vision_start
    assert vision_length % token_per_frame == 0
    current_frame_num = vision_length // token_per_frame

    if current_frame_num <= compress_frame_num:
        return past_key_values, None

    num_layers = len(model.model.language_model.layers)

    if method == "prototrack-kv":
        ks0 = past_key_values.layers[0].keys
        device0 = ks0.device; _, num_heads0, _, _ = ks0.shape
        state = _get_or_reset_prototrack_state(model=model, num_layers=num_layers, num_heads=num_heads0, max_proto_frames=max(1, int(prototrack_proto_frames)), device=device0, reset=bool(is_first_block))
        rope_theta = _infer_rope_theta(model)

    for layer_idx in range(num_layers):
        key_states = past_key_values.layers[layer_idx].keys
        value_states = past_key_values.layers[layer_idx].values
        key_states_to_compress = key_states[:, :, system_size:, :]
        value_states_to_compress = value_states[:, :, system_size:, :]

        if method == "prototrack-kv":
            # Allocate residual-statistics histograms H_k^K/H_k^V lazily once
            # the per-head dimension and token_per_frame are known.  Shape per
            # layer: [num_heads, max_proto_frames, token_per_frame, S_eff, C].
            hist_k_layer = None
            hist_v_layer = None
            if int(prototrack_pq_subspaces) > 0 and int(prototrack_pq_codebook_size) > 1:
                _, num_heads_l, _, head_dim_l = key_states_to_compress.shape
                S_eff_l = _nearest_divisor(int(head_dim_l), int(prototrack_pq_subspaces))
                C_l = int(prototrack_pq_codebook_size)
                need_hist_reset = (
                    getattr(state, "pq_hist_k", None) is None
                    or getattr(state, "pq_hist_v", None) is None
                    or len(state.pq_hist_k) != num_layers
                    or int(getattr(state, "pq_hist_token_per_frame", 0)) != int(token_per_frame)
                    or int(getattr(state, "pq_last_subspaces", 0)) not in (0, S_eff_l)
                    or int(getattr(state, "pq_last_codebook_size", 0)) not in (0, C_l)
                )
                if need_hist_reset:
                    state.pq_hist_k = [
                        torch.zeros((num_heads_l, state.max_proto_frames, int(token_per_frame), S_eff_l, C_l), device=key_states.device, dtype=torch.int32)
                        for _ in range(num_layers)
                    ]
                    state.pq_hist_v = [
                        torch.zeros((num_heads_l, state.max_proto_frames, int(token_per_frame), S_eff_l, C_l), device=key_states.device, dtype=torch.int32)
                        for _ in range(num_layers)
                    ]
                    state.pq_hist_token_per_frame = int(token_per_frame)
                    state.pq_last_subspaces = int(S_eff_l)
                    state.pq_last_codebook_size = int(C_l)
                    state.pq_codebooks_k = [None] * num_layers
                    state.pq_codebooks_v = [None] * num_layers
                hist_k_layer = state.pq_hist_k[layer_idx]
                hist_v_layer = state.pq_hist_v[layer_idx]

            new_k_vis, new_v_vis, state.initialized, p_final, cb_k, cb_v = _prototrack_kv_compress_layer(
                key_states_to_compress=key_states_to_compress,
                value_states_to_compress=value_states_to_compress,
                token_per_frame=token_per_frame,
                compress_frame_num=compress_frame_num,
                proto_frames_max=state.max_proto_frames,
                state_counts=state.counts[layer_idx],
                state_taus=(state.taus[layer_idx] if getattr(state, "taus", None) is not None else None),
                state_initialized=state.initialized,
                state_proto_frames_cur=int(state.proto_frames_cur),
                cur_frame_end=(int(prototrack_frame_end) if prototrack_frame_end is not None else None),
                rope_theta=float(rope_theta),
                pq_subspaces=int(prototrack_pq_subspaces),
                pq_codebook_size=int(prototrack_pq_codebook_size),
                pq_kmeans_iters=int(prototrack_pq_kmeans_iters),
                pq_sample_size=int(prototrack_pq_sample_size),
                pq_codebook_k=(state.pq_codebooks_k[layer_idx] if state.pq_codebooks_k is not None else None),
                pq_codebook_v=(state.pq_codebooks_v[layer_idx] if state.pq_codebooks_v is not None else None),
                pq_hist_k=hist_k_layer,
                pq_hist_v=hist_v_layer,
                pq_seed=int(prototrack_pq_seed + layer_idx * 101),
                pq_decode_top_s=int(prototrack_decode_top_s),
                pq_decode_beam_size=int(prototrack_decode_beam_size),
                pq_decode_eps=float(prototrack_decode_eps),
                # ── pass timer into compress layer ───────────────────────
                cuda_timer=cuda_timer,
                timing=timing if cuda_timer is None else None,
            )
            state.proto_frames_cur = int(p_final)
            if state.pq_codebooks_k is not None: state.pq_codebooks_k[layer_idx] = cb_k
            if state.pq_codebooks_v is not None: state.pq_codebooks_v[layer_idx] = cb_v
            if cb_k is not None:
                state.pq_last_subspaces = int(cb_k.shape[0]); state.pq_last_codebook_size = int(cb_k.shape[1])
            new_k = torch.cat([key_states[:, :, :system_size, :], new_k_vis], dim=2)
            new_v = torch.cat([value_states[:, :, :system_size, :], new_v_vis], dim=2)
            past_key_values.layers[layer_idx].keys = new_k
            past_key_values.layers[layer_idx].values = new_v
            continue

        if adaptive_pooling:
            idx = layer_idx // max(1, (num_layers // len(pooling_func_list)))
            avg_pooling_nd = pooling_func_list[idx]
            attn_pool_size = pool_size_list[idx]
        else:
            avg_pooling_nd = -1

        _, num_heads, _, _ = key_states.shape
        total_tokens_to_keep = compress_frame_num * token_per_frame

        # ------- selection-based methods (swa / uniform / infinipot-v) ------

        if method == "swa":
            start_idx = key_states_to_compress.shape[2] - total_tokens_to_keep
            all_indices = torch.arange(start_idx, key_states_to_compress.shape[2], device=key_states.device).unsqueeze(0).expand(num_heads, -1)

        elif method == "uniform":
            total_tokens = key_states_to_compress.shape[2]
            step = total_tokens / total_tokens_to_keep
            selected = (torch.arange(total_tokens_to_keep, device=key_states.device, dtype=torch.float32) * step).to(torch.long)
            all_indices = selected.unsqueeze(0).expand(num_heads, -1)

        elif method == "infinipot-v":
            attn_budget = round((1 - tar_ratio) * compress_frame_num * token_per_frame)
            if tar_ratio > 0:
                tar_budget = compress_frame_num * token_per_frame - attn_budget
                query_frame = int(current_frame_num * query_ratio)
                query_length = int(query_frame * token_per_frame)
                total_length = key_states_to_compress.shape[2]
                query_frame_emb = key_states_to_compress[:, :, -query_length:, :]
                qn = query_frame_emb / (query_frame_emb.norm(dim=-1, keepdim=True) + 1e-9)
                key_frame_emb = key_states_to_compress[:, :, :-query_length, :]
                kn = key_frame_emb / (key_frame_emb.norm(dim=-1, keepdim=True) + 1e-9)
                qn_r = qn.reshape(qn.shape[0], qn.shape[1], query_frame, token_per_frame, qn.shape[-1])
                kn_r = kn.reshape(kn.shape[0], kn.shape[1], -1, token_per_frame, kn.shape[-1])
                key_score = -(qn_r.unsqueeze(3) * kn_r.unsqueeze(2)).sum(dim=-1).mean(dim=2).reshape(qn.shape[0], qn.shape[1], -1)
                recent_indices = torch.arange(total_length-query_length, total_length).unsqueeze(0).expand(num_heads, -1).to(key_states_to_compress.device)
                selected_indices = key_score.topk(tar_budget - query_length, dim=-1).indices.squeeze(0)
                tar_indices, _ = torch.cat([selected_indices, recent_indices], dim=-1).sort(dim=-1)
            else:
                tar_indices = None
            val_norm_score = value_states_to_compress.norm(dim=-1)[0, :, :]
            if avg_pooling_nd > 0:
                pool_size = attn_pool_size
                head_num, _ = val_norm_score.shape
                frame_num = val_norm_score.shape[-1] // token_per_frame
                if avg_pooling_nd == 1:
                    avg_pool = torch.nn.AvgPool1d(kernel_size=pool_size, stride=1, padding=pool_size//2)
                elif avg_pooling_nd == 2:
                    avg_pool = torch.nn.AvgPool2d(kernel_size=(pool_size,pool_size), stride=(1,1), padding=(pool_size//2,pool_size//2))
                else:
                    avg_pool = torch.nn.AvgPool3d(kernel_size=(pool_size,pool_size,pool_size), stride=(1,1,1), padding=(pool_size//2,pool_size//2,pool_size//2))
                patch_size = getattr(model.config, "height_width", 14)
                if avg_pooling_nd > 1:
                    original_numel = val_norm_score.numel()
                    val_norm_reshaped = val_norm_score.reshape(head_num, frame_num, patch_size, -1)
                    T, Hh, Ww = val_norm_reshaped.shape[-3], val_norm_reshaped.shape[-2], val_norm_reshaped.shape[-1]
                    ps_eff = min(pool_size, T, Hh, Ww)
                    if ps_eff % 2 == 0: ps_eff -= 1
                    if ps_eff >= 2 and ps_eff != pool_size:
                        pool_size = ps_eff
                        avg_pool = torch.nn.AvgPool2d(kernel_size=(pool_size,pool_size), stride=(1,1), padding=(pool_size//2,pool_size//2)) if avg_pooling_nd == 2 else torch.nn.AvgPool3d(kernel_size=(pool_size,pool_size,pool_size), stride=(1,1,1), padding=(pool_size//2,pool_size//2,pool_size//2))
                    if ps_eff >= 2:
                        val_norm_score = avg_pool(val_norm_reshaped).reshape(head_num, -1)
                        assert original_numel == val_norm_score.numel()
                else:
                    val_norm_score = avg_pool(val_norm_score)
            if tar_indices is not None:
                hi = torch.arange(num_heads, device=key_states.device).unsqueeze(1).expand(-1, tar_indices.size(1))
                val_norm_score[hi, tar_indices] = val_norm_score.max() + 1
            all_indices, _ = val_norm_score.topk(compress_frame_num * token_per_frame, dim=-1).indices.sort(dim=-1)

        else:
            raise ValueError(f"Unknown compression method: {method}")

        batch_indices = torch.zeros_like(all_indices)
        head_indices = torch.arange(num_heads, device=key_states.device).unsqueeze(1).expand(-1, all_indices.size(1))
        new_key_states_to_compress = key_states_to_compress[batch_indices, head_indices, all_indices].unsqueeze(0)
        new_value_states_to_compress = value_states_to_compress[batch_indices, head_indices, all_indices].unsqueeze(0)
        past_key_values.layers[layer_idx].keys = torch.cat([key_states[:, :, :system_size, :], new_key_states_to_compress], dim=2)
        past_key_values.layers[layer_idx].values = torch.cat([value_states[:, :, :system_size, :], new_value_states_to_compress], dim=2)

    return past_key_values, None
