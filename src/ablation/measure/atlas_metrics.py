"""
Atlas metrics for in-memory ablation models — thin wrappers over the EXISTING pipeline.

The audit required: do NOT reimplement measurement. These functions reuse collect_residuals
/ twonn_dimension (intrinsic.py), partition_pairs / model_head_geometry / bootstrap_ci
(atlas_crossarch.py), and _layer_for_relative_depth (atlas_dhead_control.py) — the exact code
that produced every prior O_h result — applied to a model already in memory (no HF reload).

Fix S2/S3: the residual-collection window is capped at the model's own n_positions, so a model
trained at seq_len=256 is never measured with a longer sequence than it has position embeddings.

The ablation models are vanilla GPT2LMHeadModel, so GPT2Backend reads them unchanged.
"""

from __future__ import annotations

import sys
import os
import statistics

# make src/ importable whether run as a module or a script
_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from intrinsic import collect_residuals, twonn_dimension
from atlas_crossarch import partition_pairs, model_head_geometry, bootstrap_ci
from atlas_dhead_control import _layer_for_relative_depth


def _model_seqlen(model, fallback=256):
    """The model's positional capacity — measurement must not exceed it (S3)."""
    cfg = getattr(model, "config", None)
    for attr in ("n_positions", "n_ctx", "max_position_embeddings"):
        v = getattr(cfg, attr, None)
        if v:
            return int(v)
    return fallback


def layer_heads(model, ids, layer, device, n_blocks, n_points):
    """Per-head residual clouds at one layer (in-memory model). Returns (by_head, N)."""
    seq_len = _model_seqlen(model)
    res = collect_residuals(model, ids, seq_len, device, (layer,),
                            n_blocks=n_blocks, max_points=n_points + 200,
                            group_mode="query")
    keys = sorted(k for k in res if k[0] == layer)
    if not keys:
        return {}, 0
    N = min(min(res[k].shape[0] for k in keys), n_points)
    return {k: res[k][:N] for k in keys}, N


def layer_dint(model, ids, layer, device, n_blocks=10, n_points=1200):
    """Mean per-head TwoNN intrinsic dim at one layer."""
    by_head, N = layer_heads(model, ids, layer, device, n_blocks, n_points)
    if not by_head:
        return float("nan"), {}, 0
    dints = [twonn_dimension(by_head[k]) for k in sorted(by_head)]
    return statistics.mean(dints), by_head, N


def inter_overlap(by_head, n_rep, d_local=7):
    """Inter-group O_h list for these head clouds (intra-group pairs excluded)."""
    parts = partition_pairs(by_head, d_local, n_rep)
    return parts["inter"] if parts["inter"] else parts["global"]


def id_profile(model, ids, device, n_blocks=10, n_points=1200, n_profile=6):
    """Coarse per-layer ID profile at fixed relative depths (peak vs plateau)."""
    geo = model_head_geometry(model)
    n_layer = geo["n_layer"]
    rels = [i / (n_profile - 1) for i in range(n_profile)]
    seen, profile = {}, []
    for r in rels:
        Lr = _layer_for_relative_depth(n_layer, r)
        if Lr not in seen:
            d, _, _ = layer_dint(model, ids, Lr, device, n_blocks, n_points)
            seen[Lr] = d
        profile.append({"rel": round(r, 2), "layer": Lr, "d_int": seen[Lr]})
    return profile


def measure_atlas(model, ids, device, *, n_blocks=12, n_points=1200, rel_depth=0.9,
                  n_deep=3, d_local=7):
    """O_h (deepest n_deep layers, bootstrap CI) + plateau d_int (relative depth)."""
    geo = model_head_geometry(model)
    n_layer, n_rep = geo["n_layer"], geo["n_rep"]
    layers = list(range(n_layer - n_deep, n_layer))

    o_cells = []
    for L in layers:
        by_head, _ = layer_heads(model, ids, L, device, n_blocks, n_points)
        if by_head:
            o_cells.extend(inter_overlap(by_head, n_rep, d_local))
    mean_o, lo_o, hi_o, _ = bootstrap_ci(o_cells) if o_cells else (float("nan"),) * 4

    Lrel = _layer_for_relative_depth(n_layer, rel_depth)
    plateau_dint, _, _ = layer_dint(model, ids, Lrel, device, n_blocks, n_points)
    return {"deep_layers": layers, "O_h": mean_o, "O_h_ci": [lo_o, hi_o],
            "plateau_d_int": plateau_dint, "plateau_layer": Lrel}
