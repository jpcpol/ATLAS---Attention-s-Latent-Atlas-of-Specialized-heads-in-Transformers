"""
NQP-Q — Robustness of the inter-head overlap O_h (pre-submission hardening).

The paper's central claim is O_h ≈ 0.283 with a tiny cross-scale spread (0.004).
A reviewer will (correctly) ask for the *experimental error* behind that number and
whether it is an artifact of the measurement hyperparameters. This script produces
the three pieces of evidence the paper promises:

  (B) Bootstrap CI on O_h         — resample head pairs, report 95% CI. Makes the
                                     experimental error on 0.283 visible.
  (C) Sensitivity to d_local      — O_h(k) for k = 4..10. If it stays near 0.28 the
                                     choice d_local=7 is not load-bearing.
  (1) Robustness across N / depth — O_h at several sample sizes and at each of the
                                     deepest layers. Stable O_h ⇒ hard to dismiss.

All forward-only, CPU-friendly. Reuses collect_residuals (intrinsic.py).
"""

from __future__ import annotations

import sys
import statistics

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch

from intrinsic import collect_residuals, twonn_dimension


# ---------------------------------------------------------------------------
# core: per-pair overlaps (so we can bootstrap over the pair distribution)
# ---------------------------------------------------------------------------

def head_bases(res_by_head: dict, d_local: int):
    """Top-d_local SVD frame per head (columns orthonormal)."""
    bases = {}
    for k in sorted(res_by_head):
        E = res_by_head[k]
        Ec = E - E.mean(0, keepdim=True)
        _, _, Vt = torch.linalg.svd(Ec, full_matrices=False)
        bases[k] = Vt[:d_local].t()                      # [dh, d_local]
    return bases


def pairwise_overlaps(res_by_head: dict, d_local: int):
    """List of O(h_i,h_j) = mean cos(principal angles) over all unordered head pairs."""
    bases = head_bases(res_by_head, d_local)
    keys = sorted(bases)
    vals = []
    for i, ki in enumerate(keys):
        for kj in keys[i + 1:]:
            s = torch.linalg.svdvals(bases[ki].t() @ bases[kj]).clamp(0, 1)
            vals.append(s.mean().item())
    return vals


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean of `values`."""
    g = torch.Generator().manual_seed(seed)
    v = torch.tensor(values, dtype=torch.float64)
    n = v.numel()
    means = torch.empty(n_boot, dtype=torch.float64)
    for b in range(n_boot):
        idx = torch.randint(n, (n,), generator=g)
        means[b] = v[idx].mean()
    lo = torch.quantile(means, alpha / 2).item()
    hi = torch.quantile(means, 1 - alpha / 2).item()
    return v.mean().item(), lo, hi, v.std(unbiased=True).item()


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _load(model_name, device, seed):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset
    import os
    torch.manual_seed(seed)
    if str(device) == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    tok = GPT2TokenizerFast.from_pretrained(model_name); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name); model.eval()
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)
    return model, ids


def deepest_layer_heads(model, ids, layer, device, n_blocks, n_points, n_heads):
    """Collect residuals for `n_heads` heads of a single layer, capped at n_points each."""
    res = collect_residuals(model, ids, 256, device, (layer,),
                            n_blocks=n_blocks, max_points=n_points + 200)
    keys = [k for k in sorted(res) if k[0] == layer][:n_heads]
    avail = min(res[k].shape[0] for k in keys)
    N = min(avail, n_points)
    return {k: res[k][:N] for k in keys}, N


def run(model_name="gpt2", device="cpu", n_blocks=12, n_heads=8,
        n_points=1200, d_local=7, seed=42):
    print(f"[Atlas-robustness] {model_name}  (n_blocks={n_blocks}, n_heads={n_heads})")
    model, ids = _load(model_name, device, seed)
    n_layer = model.config.n_layer
    deep = list(range(n_layer - 3, n_layer))

    # reference cloud: deepest layer, full N, d_local default
    by_head, N = deepest_layer_heads(model, ids, deep[-1], device,
                                     n_blocks, n_points, n_heads)
    print(f"  reference: layer {deep[-1]}, {len(by_head)} heads, N={N}")

    # ---- (B) bootstrap CI on O_h --------------------------------------------
    pair_vals = pairwise_overlaps(by_head, d_local)
    mean, lo, hi, sd = bootstrap_ci(pair_vals)
    print(f"\n  (B) O_h bootstrap (over {len(pair_vals)} head pairs):")
    print(f"      O_h = {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (pair sd {sd:.3f})")

    # ---- (C) sensitivity to d_local -----------------------------------------
    print(f"\n  (C) sensitivity to d_local:")
    print(f"      {'k':>3} | {'O_h':>6}")
    sens = {}
    for k in range(4, 11):
        ov = statistics.mean(pairwise_overlaps(by_head, k))
        sens[k] = ov
        print(f"      {k:>3} | {ov:>6.3f}")
    spread_k = max(sens.values()) - min(sens.values())
    print(f"      d_local spread (k=4..10): {spread_k:.3f}")

    # ---- (1) robustness across N --------------------------------------------
    print(f"\n  (1a) robustness across N (deepest layer):")
    print(f"      {'N':>5} | {'O_h':>6}")
    for frac in (0.25, 0.5, 1.0):
        Nsub = max(50, int(N * frac))
        sub = {k: v[:Nsub] for k, v in by_head.items()}
        ov = statistics.mean(pairwise_overlaps(sub, d_local))
        print(f"      {Nsub:>5} | {ov:>6.3f}")

    # ---- (1) robustness across depth ----------------------------------------
    print(f"\n  (1b) robustness across depth (each of deepest 3 layers, fixed N/d_local):")
    print(f"      {'layer':>5} | {'O_h':>6} | {'dim_int':>7}")
    depth_vals = []
    for L in deep:
        bh, Nl = deepest_layer_heads(model, ids, L, device, n_blocks, n_points, n_heads)
        ov = statistics.mean(pairwise_overlaps(bh, d_local))
        di = statistics.mean(twonn_dimension(bh[k]) for k in bh)
        depth_vals.append(ov)
        print(f"      {L:>5} | {ov:>6.3f} | {di:>7.1f}")
    spread_depth = max(depth_vals) - min(depth_vals)
    print(f"      depth spread: {spread_depth:.3f}")

    print(f"\n{'='*66}\n[VERDICT] {model_name}\n{'='*66}")
    print(f"  O_h = {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  robust to d_local (spread {spread_k:.3f}) and depth (spread {spread_depth:.3f})")
    return {"model": model_name, "O_h": mean, "ci": (lo, hi), "pair_sd": sd,
            "d_local_sens": sens, "d_local_spread": spread_k,
            "depth_vals": depth_vals, "depth_spread": spread_depth, "N": N}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="NQP-Q — robustness/CI of inter-head overlap")
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--n-blocks", type=int, default=12)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-points", type=int, default=1200)
    p.add_argument("--d-local", type=int, default=7)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    run(model_name=args.model, device=args.device, n_blocks=args.n_blocks,
        n_heads=args.n_heads, n_points=args.n_points, d_local=args.d_local)
