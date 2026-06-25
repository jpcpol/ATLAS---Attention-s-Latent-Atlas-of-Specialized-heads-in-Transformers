"""
NQP-Q — Collect the numeric data behind the paper figures into one JSON.

Re-collects the per-model H×H overlap matrix (Fig 1, the iconic figure), the per-head
intrinsic vs linear dimension (Fig 3), and the d_local sweep (suppl). Scalars already
measured in prior runs (bootstrap CIs, inter-corpus, Q06) are recorded here verbatim so
the figure script has a single source of truth. Output: docs/figure_data.json
"""

from __future__ import annotations

import sys
import json
import statistics

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch

from intrinsic import twonn_dimension, linear_dim_90
from atlas_robustness import head_bases, deepest_layer_heads, _load


def overlap_matrix(by_head, d_local=7):
    """Full H×H matrix of pairwise overlaps (diagonal = 1)."""
    bases = head_bases(by_head, d_local)
    keys = sorted(bases)
    H = len(keys)
    M = [[0.0] * H for _ in range(H)]
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i == j:
                M[i][j] = 1.0
            elif j > i:
                s = torch.linalg.svdvals(bases[ki].t() @ bases[kj]).clamp(0, 1)
                v = s.mean().item()
                M[i][j] = v; M[j][i] = v
    return M, [f"H{k[1]}" for k in keys]


def collect(model_name, device="cpu", n_blocks=12, n_heads=8, n_points=1200,
            d_local=7, seed=42):
    print(f"[figure_data] {model_name}...")
    model, ids = _load(model_name, device, seed)
    deep_layer = model.config.n_layer - 1
    by_head, N = deepest_layer_heads(model, ids, deep_layer, device,
                                     n_blocks, n_points, n_heads)
    M, labels = overlap_matrix(by_head, d_local)
    # per-head intrinsic + linear dim (Fig 3)
    int_dims = [twonn_dimension(by_head[k]) for k in sorted(by_head)]
    lin_dims = [linear_dim_90(by_head[k]) for k in sorted(by_head)]
    # d_local sweep (suppl)
    sweep = {}
    for k in range(4, 11):
        bM, _ = overlap_matrix(by_head, k)
        off = [bM[i][j] for i in range(len(bM)) for j in range(len(bM)) if i != j]
        sweep[k] = statistics.mean(off)
    print(f"  layer {deep_layer}, N={N}, mean off-diag O_h={statistics.mean([M[i][j] for i in range(len(M)) for j in range(len(M)) if i!=j]):.3f}")
    return {"model": model_name, "n_layer": model.config.n_layer, "layer": deep_layer,
            "N": N, "labels": labels, "overlap_matrix": M,
            "intrinsic_dims": int_dims, "linear_dims": lin_dims,
            "dlocal_sweep": sweep}


def main():
    out = {"models": {}}
    for m in ("gpt2", "gpt2-medium", "gpt2-large"):
        out["models"][m] = collect(m)

    # --- scalars measured in prior runs (single source of truth) -------------
    # bootstrap O_h + 95% CI (atlas_robustness.py)
    out["overlap_ci"] = {
        "gpt2":        {"O_h": 0.284, "lo": 0.276, "hi": 0.292, "params_M": 124},
        "gpt2-medium": {"O_h": 0.277, "lo": 0.267, "hi": 0.288, "params_M": 355},
        "gpt2-large":  {"O_h": 0.281, "lo": 0.272, "hi": 0.290, "params_M": 774},
    }
    # inter-corpus (atlas_intercorpus.py, gpt2)
    out["intercorpus"] = {
        "wikitext": {"O_h": 0.284, "lo": 0.276, "hi": 0.292},
        "c4":       {"O_h": 0.277, "lo": 0.269, "hi": 0.285},
    }
    # Q06 negative (autoencoder.py)
    out["q06"] = {"top1_damage_ppl": 16.45, "pca_rank7_recovered": 55.9,
                  "ae_64_7_64_recovered": 55.8, "fvu": 0.38}

    with open("../docs/figure_data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n[figure_data] wrote docs/figure_data.json")


if __name__ == "__main__":
    main()
