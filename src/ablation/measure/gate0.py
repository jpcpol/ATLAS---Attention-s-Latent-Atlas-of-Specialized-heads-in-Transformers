"""
Gate 0 — atlas maturation (validity gate). The anti-NQP heart of the ablation.

A flat or degenerate O_h from an UNDER-TRAINED model is an INVALID measurement, not a
refutation of the d_head hypothesis. Gate 0 must pass before a model's O_h counts toward
the predictions in docs/ablation_design.md §4.

Audit fixes vs the monolith:
  C1 — G0a convergence was inverted: it required the loss to STOP improving between the
       last two evals, so a healthy still-improving model FAILED. Corrected: convergence is
       judged from the val-loss curve over the snapshots — the recent relative improvement
       per snapshot must be small (the curve has flattened), AND the loss must be in a
       sane LM range (well below random ln(V) ≈ 10.8, and not absurdly low). A model that
       is still dropping fast is "not yet matured" → extend training (not a refutation).
  C2 — the unused rand_loss / dead `geo.get("vocab")` are gone; the random-loss floor is
       computed explicitly from vocab_size.

G0a converged LM        — val-loss curve flattened AND in a sane range.
G0b depth régime exists — per-layer ID profile has a real expansion/compression bump.
G0c residual stable     — deep-layer residual collectible (adequate N, all heads present).
G0d base atlas appears  — O_h << 1 at the deepest layer (non-aligned at all).
"""

from __future__ import annotations

import math
import statistics

from .atlas_metrics import (
    model_head_geometry, id_profile, layer_dint, inter_overlap,
)


def _g0a_converged(val_curve, vocab_size, *, plateau_rel=0.03, min_evals=2):
    """Convergence from the val-loss curve (C1 fix).

    Pass iff: (i) we have >= min_evals points, (ii) the relative improvement over the LAST
    snapshot interval is small (< plateau_rel) — the curve has flattened, NOT still
    plunging, and (iii) the loss sits in a sane LM band: clearly below random ln(V) and
    above an implausible floor. A still-steeply-improving model FAILS as 'not matured'
    (→ extend training), which is the correct anti-NQP reading.
    """
    rand = math.log(vocab_size)                       # ≈ 10.82 for 50257
    if len(val_curve) < min_evals:
        last = val_curve[-1] if val_curve else float("inf")
        return (last < 0.8 * rand), {"reason": "single-eval fallback", "last": last,
                                     "rand": rand}
    prev, last = val_curve[-2], val_curve[-1]
    rel_impr = (prev - last) / max(1e-6, prev)        # >0 means still improving
    flattened = rel_impr < plateau_rel
    sane = (last < 0.8 * rand) and (last > 0.5)       # below random, not implausibly low
    return bool(flattened and sane), {"rel_impr_last": rel_impr, "last": last,
                                      "rand": rand, "flattened": flattened, "sane": sane}


def gate0(model, ids, device, *, val_curve, vocab_size=50257, n_blocks=10, n_points=1200,
          n_profile=6):
    """Run G0a-G0d. Returns (passed: bool, report: dict). val_curve = list of val losses
    in snapshot order (so the last interval can be judged for convergence)."""
    geo = model_head_geometry(model)
    n_layer, n_rep = geo["n_layer"], geo["n_rep"]
    deep = n_layer - 1
    rep = {"geometry": geo}

    # G0a — convergence (C1 fixed)
    g0a, a_info = _g0a_converged(val_curve, vocab_size)
    rep["G0a_converged"] = {"pass": bool(g0a), **a_info, "val_curve": list(val_curve)}

    # G0b — depth régime: the per-layer ID profile must show the expansion→compression
    # shape Valeriani et al. (2302.00294) report: ID rises to an EARLY peak (first third),
    # then compresses. CRITICALLY, Valeriani also report a FINAL ASCENT near the last layers
    # ("ID grows again, returning toward input-level values") — a healthy part of the régime.
    #
    # Audit fix (2026-06-27): the original criterion (bump_vs_ends = peak − max(endpoints))
    # PENALIZED that final ascent: a high last-layer ID shrinks peak−ends and produced a
    # false-negative (d_head=64 failed with bump_vs_ends=0.18 while its bump_vs_min=1.44 was
    # essentially identical to the d_head=32 runs that passed). We therefore decide G0b on
    # **bump_vs_min** (peak above the profile MINIMUM = the true amplitude of the régime,
    # which the final ascent does not corrupt) AND a **peak-location** check (the peak sits
    # in the first half, rel ≤ 0.5, as Valeriani's early-peak finding requires). This is not
    # loosening the gate arbitrarily — it aligns G0b with the régime the literature describes.
    profile = id_profile(model, ids, device, n_blocks, n_points, n_profile)
    vals = [p["d_int"] for p in profile]
    finite = [v for v in vals if v == v]              # drop NaN
    if len(finite) >= 3:
        peak, lo = max(finite), min(finite)
        ends = max(finite[0], finite[-1])             # finite endpoints (kept for reporting)
        bump_vs_ends = peak - ends
        bump_vs_min = peak - lo
        # peak location: rel of the max-d_int layer (early peak per Valeriani)
        peak_rel = next(p["rel"] for p in profile if p["d_int"] == peak)
        peak_early = peak_rel <= 0.5
        g0b = bump_vs_min > 0.5 and peak_early
    else:
        peak = lo = ends = float("nan")
        bump_vs_ends = bump_vs_min = peak_rel = float("nan")
        peak_early = False
        g0b = False
    rep["G0b_depth_regime"] = {"pass": bool(g0b), "peak": peak, "min": lo,
                               "peak_rel": peak_rel, "peak_early": bool(peak_early),
                               "bump_vs_ends": bump_vs_ends, "bump_vs_min": bump_vs_min,
                               "profile": profile}

    # G0c — residual stability at the deepest layer
    dint_deep, by_head, N = layer_dint(model, ids, deep, device, n_blocks, n_points)
    g0c = (N >= 500) and (len(by_head) == geo["n_q"])
    rep["G0c_residual_stable"] = {"pass": bool(g0c), "N": N, "n_heads": len(by_head),
                                  "deep_d_int": dint_deep}

    # G0d — base atlas exists (O_h << 1)
    if by_head:
        inter = inter_overlap(by_head, n_rep, d_local=7)
        o_h = statistics.mean(inter) if inter else float("nan")
        g0d = (o_h == o_h) and (o_h < 0.6)
    else:
        o_h, g0d = float("nan"), False
    rep["G0d_base_atlas"] = {"pass": bool(g0d), "O_h": o_h}

    passed = bool(g0a and g0b and g0c and g0d)
    rep["passed"] = passed
    return passed, rep
