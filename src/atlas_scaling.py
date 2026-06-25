"""
NQP-Q — Atlas validity across model scale (EXP-Q05d-scale).

Q05d found head-specific manifolds with INCOMPATIBLE coordinates in GPT-2 small
(inter-head subspace overlap ≈ 0.28). Q04 already showed the intrinsic dimension
(~7) is scale-invariant. This closes the remaining flank: does the NON-ALIGNMENT
(the atlas structure itself) also hold at medium/large, or is it a small-model
artifact? We re-measure the inter-head subspace overlap and head-centered pooled
dim across {small, medium, large}, controlled (same N, same #heads, deepest layers).

If overlap stays low (~0.3) at all scales ⇒ the non-aligned manifold atlas is a
robust property within the GPT-2 family, matching the scale-invariance of dim(M_h).
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
from manifold import atlas_test


def atlas_profile(model_name, device="cpu", n_blocks=8, n_points=1200,
                  n_heads=8, d_local=7, seed=42):
    """Inter-head subspace overlap + head-centered pooled dim for the deepest 3 layers."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset
    import os
    torch.manual_seed(seed)
    if str(device) == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    tok = GPT2TokenizerFast.from_pretrained(model_name); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name); model.eval()
    n_layer = model.config.n_layer
    deep = tuple(range(n_layer - 3, n_layer))            # deepest 3 layers
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)

    res = collect_residuals(model, ids, 256, device, deep, n_blocks=n_blocks,
                            max_points=n_points + 300)
    # restrict to a FIXED set of (layer, head) keys and equal N for comparability
    keys = sorted(res.keys())
    avail = min(res[k].shape[0] for k in keys)
    N = min(avail, n_points)
    # sample n_heads keys from the deepest layer for the overlap (same #heads/model)
    deep_keys = [k for k in keys if k[0] == deep[-1]][:n_heads]
    by_head = {k: res[k][:N] for k in deep_keys}
    print(f"    (controlled: N={N}, {len(by_head)} heads from layer {deep[-1]}, d_local={d_local})")
    overlap, centered_dim = atlas_test(by_head, d_local=d_local)
    # also per-head intrinsic dim for context
    ints = [twonn_dimension(by_head[k]) for k in by_head]
    return {"model": model_name, "n_layer": n_layer, "overlap": overlap,
            "centered_dim": centered_dim, "dim_int": statistics.mean(ints)}


def run(models=("gpt2", "gpt2-medium", "gpt2-large"), device="cpu", n_blocks=8):
    results = []
    for m in models:
        print(f"\n[Atlas-scale] Profiling {m}...")
        r = atlas_profile(m, device=device, n_blocks=n_blocks)
        results.append(r)
        print(f"  {m}: overlap={r['overlap']:.3f}  centered_dim={r['centered_dim']:.1f}  "
              f"dim_int={r['dim_int']:.1f}")

    print(f"\n{'='*70}\n[Atlas-scale VERDICT] is non-alignment scale-invariant?\n{'='*70}")
    print(f"  {'model':>14} | {'overlap':>7} | {'centered_dim':>12} | {'dim_int':>7}")
    for r in results:
        print(f"  {r['model']:>14} | {r['overlap']:>7.3f} | {r['centered_dim']:>12.1f} | "
              f"{r['dim_int']:>7.1f}")
    overlaps = [r["overlap"] for r in results]
    spread = max(overlaps) - min(overlaps)
    all_low = all(o < 0.45 for o in overlaps)
    print(f"\n  overlap spread across scale: {spread:.3f}")
    if all_low and spread < 0.2:
        print(f"  => ATLAS IS SCALE-INVARIANT: heads remain non-aligned (overlap "
              f"{min(overlaps):.2f}-{max(overlaps):.2f}) at all sizes. The non-aligned manifold "
              f"atlas is a robust property within the GPT-2 family, matching dim(M_h) invariance.")
    elif all_low:
        print(f"  => atlas holds but overlap varies somewhat ({spread:.2f}); non-alignment robust, "
              f"degree scale-dependent.")
    else:
        print(f"  => atlas WEAKENS at scale (some overlap ≥0.45); coordinate sharing grows with "
              f"size — non-alignment may be a small-model property.")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="NQP-Q — atlas non-alignment across scale")
    p.add_argument("--models", type=str, nargs="+",
                   default=["gpt2", "gpt2-medium", "gpt2-large"])
    p.add_argument("--n-blocks", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    run(models=tuple(args.models), n_blocks=args.n_blocks, device=args.device)
