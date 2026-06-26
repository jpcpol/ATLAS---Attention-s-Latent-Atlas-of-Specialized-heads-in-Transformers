"""
ATLAS — Intra-model control: does d_int drive O_h, or is d_head the driver?

The cross-architecture result (§3.1b) gives two clusters: d_head 64 -> O_h ~ 0.28,
d_head 128 -> O_h ~ 0.20. But d_head is correlated *between* models with several other
quantities — intrinsic dimension (TwoNN), d_model, n_KV. The most dangerous confounder
is intrinsic dimension: the same data show d_int and O_h move together across models
(d_int 7.1 -> O_h 0.29; d_int 9-11 -> O_h 0.20), so the causal chain could be

        d_head -> d_int -> O_h        (d_head is only a proxy)
   or   d_head ------------> O_h       (d_head is the driver)

and the four between-model points cannot tell them apart.

THIS CONTROL discriminates them WITHOUT retraining, by looking *inside a single model*:

  For each query head h in a layer, measure
     d_int_h  = TwoNN of that head's residual cloud, and
     Obar_h   = mean inter-group overlap of head h with all *other* heads,
  then correlate d_int_h against Obar_h across the heads of that model (Spearman).

Interpretation:
  * STRONG negative within-model correlation (heads with higher d_int have lower overlap)
    => the d_int -> O_h relation also lives *inside* a model. d_int becomes a serious
       candidate driver and d_head may be only its proxy. The d_head ablation should be
       reframed to vary d_int explicitly, or at least to disentangle the two.
  * NO within-model correlation, even though it exists *between* models
    => the d_int <-> O_h link is an architecture-level (between-model) phenomenon, not a
       per-head one. This is the outcome that *protects* the d_head lead: it says d_head
       (a fixed architectural number) organizes the heads, and d_int is a downstream
       readout that only varies with architecture, not head-by-head.

The point is not to "prove" d_head; it is to find out, cheaply, whether the d_int
confounder is real before spending compute on a matched-scale ablation. Per ChatGPT's
cross-check (its "favourite" control) and our anti-NQP discipline.

Spearman + a permutation p-value (no scipy dependency). Averaged over the deepest layers
so the result is not a single-layer artifact. Forward-only, CPU-friendly.
"""

from __future__ import annotations

import sys
import json
import random
import argparse
import statistics

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch

from intrinsic import twonn_dimension
from atlas_robustness import head_bases
from atlas_crossarch import (
    _load_any, _deep_query_heads, kv_group_of, model_head_geometry,
)


# ---------------------------------------------------------------------------
# per-head mean inter-group overlap (one scalar Obar_h per head)
# ---------------------------------------------------------------------------

def per_head_inter_overlap(res_by_head: dict, d_local: int, n_rep: int):
    """For each head, its mean overlap with all OTHER inter-group heads.

    Returns {head_index: Obar_h}. Intra-group partners (shared KV/value space) are
    excluded, exactly as in the cross-arch table, so the per-head number is the honest
    one. With MHA (n_rep == 1) every other head is inter-group.
    """
    bases = head_bases(res_by_head, d_local)
    keys = sorted(bases)                                   # [(layer, head), ...]
    # full pairwise overlap matrix, then average the inter-group entries per head
    sums = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}
    for i, ki in enumerate(keys):
        for kj in keys[i + 1:]:
            if kv_group_of(ki[1], n_rep) == kv_group_of(kj[1], n_rep):
                continue                                   # skip intra-group pairs
            s = torch.linalg.svdvals(bases[ki].t() @ bases[kj]).clamp(0, 1)
            o = s.mean().item()
            sums[ki] += o; counts[ki] += 1
            sums[kj] += o; counts[kj] += 1
    return {k[1]: (sums[k] / counts[k]) for k in keys if counts[k] > 0}


# ---------------------------------------------------------------------------
# Spearman rho + permutation p-value (no scipy)
# ---------------------------------------------------------------------------

def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0                          # average rank, 1-based
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def _pearson(a, b):
    n = len(a)
    ma = sum(a) / n; mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((a[i] - ma) ** 2 for i in range(n))
    vb = sum((b[i] - mb) ** 2 for i in range(n))
    if va == 0 or vb == 0:
        return 0.0
    return cov / (va ** 0.5 * vb ** 0.5)


def spearman(x, y, n_perm=20000, seed=0):
    """Spearman rho + two-sided permutation p-value over y-label shuffles."""
    rx, ry = _rank(x), _rank(y)
    rho = _pearson(rx, ry)
    rng = random.Random(seed)
    ry_shuf = list(ry)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(ry_shuf)
        if abs(_pearson(rx, ry_shuf)) >= abs(rho) - 1e-12:
            hits += 1
    p = (hits + 1) / (n_perm + 1)
    return rho, p


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(model_name="gpt2", device="cpu", n_blocks=12, n_points=1200,
        n_deep=3, d_local=7, seed=42, offload=False, n_perm=20000):
    """Within-model d_int_h vs Obar_h correlation, pooled over the deepest n_deep layers.

    Pooling: each (layer, head) contributes one (d_int_h, Obar_h) point. d_local is fixed
    (default 7, matching the cross-arch control) so the per-head overlap is measured on a
    common frame size across heads and models.
    """
    print(f"[Atlas-intramodel] {model_name}" + ("  [offload]" if offload else ""))
    model, ids = _load_any(model_name, device, seed, offload=offload)
    geo = model_head_geometry(model)
    n_rep = geo["n_rep"]
    layers = list(range(geo["n_layer"] - n_deep, geo["n_layer"]))
    print(f"  geometry: family={geo['family']} n_q={geo['n_q']} n_kv={geo['n_kv']} "
          f"n_rep={n_rep} d_head={geo['d_head']} | layers={layers} d_local={d_local}")

    dints, obars, tags = [], [], []
    print(f"\n  per-head points (d_int_h, Obar_h), pooled over {n_deep} deep layers:")
    for L in layers:
        by_head, N = _deep_query_heads(model, ids, L, device, n_blocks, n_points)
        # per-head intrinsic dim (TwoNN) and per-head mean inter-group overlap
        dh = {k[1]: twonn_dimension(by_head[k]) for k in sorted(by_head)}
        ob = per_head_inter_overlap(by_head, d_local, n_rep)
        for h in sorted(ob):
            dints.append(dh[h]); obars.append(ob[h]); tags.append((L, h))
        rho_L, _ = spearman([dh[h] for h in sorted(ob)],
                            [ob[h] for h in sorted(ob)], n_perm=2000, seed=L)
        print(f"      layer {L}: {len(ob)} heads, N={N}, "
              f"d_int {min(dh.values()):.1f}-{max(dh.values()):.1f}, "
              f"Obar {min(ob.values()):.3f}-{max(ob.values()):.3f}, "
              f"per-layer rho={rho_L:+.3f}")

    rho, p = spearman(dints, obars, n_perm=n_perm, seed=seed)
    n = len(dints)
    print(f"\n{'='*70}\n[INTRA-MODEL CONTROL] {model_name}\n{'='*70}")
    print(f"  pooled n = {n} head-points over layers {layers}")
    print(f"  Spearman rho(d_int_h, Obar_h) = {rho:+.3f}   permutation p = {p:.4f}")

    # verdict — does the d_int -> O_h relation exist INSIDE a model?
    SIG = 0.05
    STRONG = 0.30                                          # |rho| we'd call non-trivial
    if p < SIG and rho <= -STRONG:
        verdict = "DINT-DRIVEN"
        msg = ("heads with higher intrinsic dim DO have lower overlap WITHIN this model. "
               "The d_int->O_h link is per-head, not only between-architecture: d_int is a "
               "serious confounder for the d_head lead. Reframe the ablation to disentangle "
               "d_head from d_int.")
    elif p >= SIG or abs(rho) < STRONG:
        verdict = "NO-INTRA-LINK"
        msg = ("no meaningful within-model d_int<->O_h correlation. The between-model "
               "d_int<->O_h relation is an architecture-level effect, not a per-head one. "
               "This PROTECTS the d_head lead: d_head (fixed per architecture) organizes "
               "the heads; per-head d_int does not predict per-head overlap.")
    else:
        verdict = "POSITIVE-LINK"
        msg = ("within-model correlation is positive (higher d_int -> HIGHER overlap), "
               "opposite to the between-model sign. The between/within signs disagree => "
               "the between-model d_int<->O_h relation is almost certainly a confound, not "
               "causal. Inspect before any d_int-based claim.")
    print(f"  => {verdict}: {msg}")

    return {"model": model_name, "geometry": geo, "layers": layers, "d_local": d_local,
            "n_points": n, "rho": rho, "p": p, "verdict": verdict,
            "points": [{"layer": t[0], "head": t[1], "d_int": d, "obar": o}
                       for t, d, o in zip(tags, dints, obars)]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="ATLAS intra-model control: d_int_h vs per-head inter-group overlap")
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--n-blocks", type=int, default=12)
    p.add_argument("--n-points", type=int, default=1200)
    p.add_argument("--n-deep", type=int, default=3)
    p.add_argument("--d-local", type=int, default=7)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--offload", action="store_true")
    p.add_argument("--out", type=str, default="", help="optional JSON dump path")
    args = p.parse_args()
    r = run(model_name=args.model, device=args.device, n_blocks=args.n_blocks,
            n_points=args.n_points, n_deep=args.n_deep, d_local=args.d_local,
            offload=args.offload)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"\n  wrote {args.out}")
