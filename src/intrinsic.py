"""
NQP-Q — Intrinsic dimension of the attention residual (EXP-Q05, Dir-D).

Q03 found the residual ε needs ~21/64 LINEAR (PCA) dims for 90% variance → not
low-rank linearly. But PCA only measures linear rank. The residual could still lie
on a low-dimensional NON-LINEAR manifold (J.P. Chancay's Dir-D). If the intrinsic
dimension d_int ≪ 21, the Q03 conclusion changes from "irreducible" to "non-linear
but compressible" — a materially different claim. This experiment measures d_int to
qualify the negative result honestly.

Estimator: TwoNN (Facco et al. 2017, Sci. Rep.). For each point, r1,r2 = distances
to its 1st and 2nd nearest neighbors; μ = r2/r1. Then μ ~ Pareto(d), and
  d = (N-1) / Σ log μ_i      (MLE of the intrinsic dimension)
Robust, parameter-free, no external deps. We cross-check with the linear PCA dim.
"""

from __future__ import annotations

import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from torch import Tensor


def twonn_dimension(X: Tensor, discard_frac: float = 0.1) -> float:
    """
    TwoNN intrinsic-dimension estimate for points X [N, D].

    Discards the top `discard_frac` of μ ratios (outliers / density inhomogeneity),
    the standard robustification from the original paper.
    """
    N = X.shape[0]
    # pairwise distances (N small enough: a few thousand × 64 is fine)
    d = torch.cdist(X, X)                                   # [N, N]
    d.fill_diagonal_(float("inf"))
    # two nearest neighbor distances
    knn = d.topk(2, dim=-1, largest=False).values          # [N, 2]
    r1, r2 = knn[:, 0], knn[:, 1]
    valid = (r1 > 1e-9)
    mu = (r2[valid] / r1[valid]).clamp(min=1.0 + 1e-9)
    logmu = mu.log().sort().values
    # discard top fraction (largest μ) as outliers
    keep = int(len(logmu) * (1.0 - discard_frac))
    logmu = logmu[:keep]
    d_int = (logmu.numel()) / logmu.sum().item()
    return d_int


def linear_dim_90(X: Tensor) -> int:
    """PCA linear dimension for 90% variance (cross-check vs Q03)."""
    Xc = X - X.mean(0, keepdim=True)
    cov = Xc.t() @ Xc / max(X.shape[0], 1)
    ev = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0)
    cum = ev.cumsum(0) / ev.sum().clamp(min=1e-12)
    return int((cum < 0.90).sum().item()) + 1


def collect_residuals(model, ids, seq_len, device, layers, n_blocks=4, max_points=3000):
    """Collect attention residual vectors ε per (layer,head). Returns dict→[N,dh]."""
    n_head = model.config.n_head
    bufs = {}
    handles = []

    def make_hook(li):
        def hook(module, inp, out):
            hidden = inp[0] if isinstance(inp, tuple) else inp
            B, T, D = hidden.shape
            dh = D // n_head
            qkv = module.c_attn(hidden)
            q, k, v = qkv.split(D, dim=2)
            shp = lambda x: x.view(B, T, n_head, dh).permute(0, 2, 1, 3)
            q, k, v = shp(q), shp(k), shp(v)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            mask = torch.tril(torch.ones(T, T, device=hidden.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
            p = scores.softmax(dim=-1)
            ctx = p @ v
            i_star = p.argmax(dim=-1)
            v_star = torch.gather(v, 2, i_star.unsqueeze(-1).expand(-1, -1, -1, dh))
            eps = (ctx - v_star)[0]                          # [H,T,dh]
            for h in range(n_head):
                bufs.setdefault((li, h), []).append(eps[h].detach())
        return hook

    for li in layers:
        handles.append(model.transformer.h[li].attn.register_forward_hook(make_hook(li)))
    nb = min(n_blocks, ids.numel() // seq_len)
    with torch.no_grad():
        for b in range(nb):
            blk = ids[b*seq_len:(b+1)*seq_len].unsqueeze(0).to(device)
            model(input_ids=blk)
    for hd in handles:
        hd.remove()
    out = {}
    for key, chunks in bufs.items():
        E = torch.cat(chunks, dim=0)
        if E.shape[0] > max_points:
            idx = torch.randperm(E.shape[0])[:max_points]
            E = E[idx]
        out[key] = E
    return out


def run_exp_q05(layers=(9, 10, 11), n_blocks=4, device="cpu", seed=42):
    """EXP-Q05 — intrinsic vs linear dimension of the residual ε."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset
    import os
    torch.manual_seed(seed)
    if str(device) == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    print("[EXP-Q05] Loading GPT-2 small...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2"); model.eval()
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)
    d_head = model.config.n_embd // model.config.n_head

    print(f"[EXP-Q05] Collecting residuals for layers {layers}...")
    res = collect_residuals(model, ids, 256, device, layers, n_blocks=n_blocks)

    print(f"\n  {'layer/head':>12} | {'N':>5} | {'PCA dim(90%)':>12} | {'intrinsic (TwoNN)':>17}")
    lin_dims, int_dims = [], []
    for key in sorted(res):
        E = res[key]
        ld = linear_dim_90(E)
        idim = twonn_dimension(E)
        lin_dims.append(ld); int_dims.append(idim)
        if key[1] == 0:  # head 0 per layer for brevity
            print(f"  L{key[0]:2d}H{key[1]:2d}      | {E.shape[0]:>5} | {ld:>12} | {idim:>17.2f}")

    import statistics
    mln = statistics.mean(lin_dims)
    mint = statistics.mean(int_dims)
    print(f"\n{'='*66}\n[EXP-Q05 VERDICT] linear vs intrinsic dimension of ε (d_head={d_head})\n{'='*66}")
    print(f"  mean PCA linear dim (90% var): {mln:.1f} / {d_head}")
    print(f"  mean intrinsic dim (TwoNN):    {mint:.1f} / {d_head}")
    ratio = mint / max(mln, 1e-9)
    if mint < 0.5 * mln and mint < 8:
        print(f"  => NON-LINEAR STRUCTURE: intrinsic ({mint:.1f}) ≪ linear ({mln:.1f}). "
              f"Q03's 'irreducible' is LINEAR-only; ε lies on a low-dim manifold. "
              f"Non-linear compression may work. Reframes the negative result.")
    else:
        print(f"  => GENUINELY HIGH-DIM: intrinsic ({mint:.1f}) ≈ linear ({mln:.1f}). "
              f"ε is not on a low-dim manifold either. Q03's irreducibility is fundamental, "
              f"not an artifact of linearity. Strengthens the negative result.")
    return {"mean_linear": mln, "mean_intrinsic": mint, "ratio": ratio}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="NQP-Q EXP-Q05 — intrinsic dimension of residual")
    p.add_argument("--layers", type=int, nargs="+", default=[9, 10, 11])
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    run_exp_q05(layers=tuple(args.layers), n_blocks=args.n_blocks, device=args.device)
