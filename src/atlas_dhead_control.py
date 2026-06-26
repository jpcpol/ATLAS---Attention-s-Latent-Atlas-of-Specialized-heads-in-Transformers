"""
ATLAS — The "cheap experiment" (no retraining): does d_int move O_h with d_head FIXED?

The cross-architecture result clustered O_h by d_head (64 -> ~0.28, 128 -> ~0.20), but the
intra-model control (atlas_intramodel.py) showed d_head is CONFOUNDED with intrinsic
dimension d_int, which predicts overlap head-by-head in Qwen (rho=-0.53). We cannot vary
d_int directly — it is an EMERGENT property of the trained model, not a hyperparameter
(Valeriani et al. 2023, arXiv:2302.00294: "ID structure is not architecturally
predetermined but emerges through self-supervised objectives"). So the clean discriminator,
suggested by the adversarial cross-check, is the converse intervention:

    hold d_head FIXED (= 64, the whole GPT-2 family) and let d_int vary via SIZE,
    then ask whether O_h tracks d_int.

  * If d_int varies across the family (d_head fixed) AND O_h tracks it
        => evidence for d_int -> O_h (d_head was a proxy; the lead should move to d_int).
  * If d_int is ~flat across the family and O_h is ~flat too
        => d_head fixed -> O_h fixed: consistent with d_head being the organizing scale,
           and d_int a downstream readout that only moves WITH d_head (between clusters).

TWO design corrections forced by the literature (so this is not a spurious comparison):

  (1) RELATIVE depth, not absolute. The per-layer ID profile has three phases
      (expansion -> compression -> ascent) whose location is set by RELATIVE depth
      (Valeriani et al.). The GPT-2 family has different depths (12/24/36/48), so the
      "deepest layer" lands in a different phase per model. We therefore sample a FIXED
      relative depth (default 0.9) across models, mapping it to each model's layer index.

  (2) Report the per-layer ID PROFILE, not just a deep-layer mean. Valeriani find the ID
      "peak grows with model size, plateau is ~constant". Collapsing ID to one scalar
      hides exactly the peak-vs-plateau structure that any latent geometric quantity
      ("Lambda") would live in. We print the profile and the plateau (relative-depth)
      value so peak and plateau are not conflated.

Forward-only, GPT-2 family (d_head=64 for all), CPU-friendly. Reuses collect_residuals,
twonn_dimension, pairwise_overlaps.
"""

from __future__ import annotations

import sys
import json
import argparse
import statistics

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch

from intrinsic import collect_residuals, twonn_dimension
from atlas_robustness import pairwise_overlaps


GPT2_DHEAD = 64        # constant across the whole family — the controlled variable


def _layer_for_relative_depth(n_layer: int, rel: float) -> int:
    """Map a relative depth in [0,1] to a 0-based layer index for an n_layer model."""
    return max(0, min(n_layer - 1, round(rel * (n_layer - 1))))


def _load_gpt2(model_name, device, seed):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset
    import os
    torch.manual_seed(seed)
    if str(device) == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    tok = GPT2TokenizerFast.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            ds = load_dataset(repo, "wikitext-103-raw-v1"); break
        except Exception:
            ds = None
    if ds is None:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)
    return model, ids


def _layer_dint(model, ids, layer, device, n_blocks, n_points):
    """Mean per-head TwoNN intrinsic dim at one layer (head-resolved residual clouds)."""
    res = collect_residuals(model, ids, 256, device, (layer,),
                            n_blocks=n_blocks, max_points=n_points + 200)
    keys = sorted(k for k in res if k[0] == layer)
    N = min(min(res[k].shape[0] for k in keys), n_points)
    by_head = {k: res[k][:N] for k in keys}
    dints = [twonn_dimension(by_head[k]) for k in keys]
    return statistics.mean(dints), by_head, N


def profile_model(model_name, device="cpu", n_blocks=10, n_points=1200,
                  rel_depth=0.9, n_profile=6, seed=42):
    """For one GPT-2 model: O_h and d_int at FIXED relative depth, plus a coarse
    per-layer d_int profile (to separate peak from plateau)."""
    model, ids = _load_gpt2(model_name, device, seed)
    n_layer = model.config.n_layer
    d_head = model.config.n_embd // model.config.n_head
    assert d_head == GPT2_DHEAD, (model_name, d_head)            # family invariant
    L = _layer_for_relative_depth(n_layer, rel_depth)

    # --- the controlled measurement: O_h and d_int at fixed relative depth ---
    dint_L, by_head, N = _layer_dint(model, ids, L, device, n_blocks, n_points)
    o_vals = pairwise_overlaps(by_head, d_local=7)
    o_h = statistics.mean(o_vals)

    # --- coarse ID profile across relative depth (peak vs plateau) ---
    rels = [i / (n_profile - 1) for i in range(n_profile)]
    seen, profile = {}, []
    for r in rels:
        Lr = _layer_for_relative_depth(n_layer, r)
        if Lr in seen:
            profile.append((round(r, 2), Lr, seen[Lr])); continue
        d, _, _ = _layer_dint(model, ids, Lr, device, n_blocks, n_points)
        seen[Lr] = d
        profile.append((round(r, 2), Lr, d))

    peak = max(p[2] for p in profile)
    return {"model": model_name, "n_layer": n_layer, "d_head": d_head,
            "rel_depth": rel_depth, "layer_at_rel": L, "N": N,
            "O_h": o_h, "d_int_at_rel": dint_L, "d_int_peak": peak,
            "profile": [{"rel": r, "layer": Lr, "d_int": d} for r, Lr, d in profile]}


def run(models=("gpt2", "gpt2-medium", "gpt2-large"), device="cpu",
        n_blocks=10, n_points=1200, rel_depth=0.9, out=""):
    print(f"[Atlas d_head control] family=GPT-2 (d_head={GPT2_DHEAD} FIXED), "
          f"rel_depth={rel_depth}\n")
    rows = []
    for m in models:
        print(f"  profiling {m} ...")
        r = profile_model(m, device=device, n_blocks=n_blocks,
                          n_points=n_points, rel_depth=rel_depth)
        rows.append(r)
        prof = "  ".join(f"{p['rel']:.1f}:{p['d_int']:.1f}" for p in r["profile"])
        print(f"    {m}: layers={r['n_layer']}  O_h={r['O_h']:.3f}  "
              f"d_int@{rel_depth}={r['d_int_at_rel']:.2f}  d_int_peak={r['d_int_peak']:.2f}")
        print(f"      ID profile (rel:d_int): {prof}")

    print(f"\n{'='*72}\n[d_head CONTROL] does d_int move O_h with d_head FIXED at 64?\n{'='*72}")
    print(f"  {'model':>12} | {'layers':>6} | {'O_h':>6} | {'d_int@rel':>9} | {'d_int_peak':>10}")
    for r in rows:
        print(f"  {r['model']:>12} | {r['n_layer']:>6} | {r['O_h']:>6.3f} | "
              f"{r['d_int_at_rel']:>9.2f} | {r['d_int_peak']:>10.2f}")

    dints = [r["d_int_at_rel"] for r in rows]
    ohs = [r["O_h"] for r in rows]
    peaks = [r["d_int_peak"] for r in rows]
    dint_spread = max(dints) - min(dints)
    oh_spread = max(ohs) - min(ohs)
    peak_spread = max(peaks) - min(peaks)
    print(f"\n  spreads across the family (d_head fixed): "
          f"d_int@rel {dint_spread:.2f}, O_h {oh_spread:.3f}, d_int_peak {peak_spread:.2f}")

    # verdict — interpret the cheap experiment
    DINT_MOVES = 1.5          # TwoNN units we'd call a real move in plateau d_int
    OH_MOVES = 0.04           # O_h units ~ half the cross-arch gap (0.08)
    if dint_spread >= DINT_MOVES and oh_spread >= OH_MOVES:
        verdict = ("D_INT->O_h SUPPORTED: with d_head fixed, plateau d_int varies and O_h "
                   "tracks it. d_int is a live driver; the d_head lead should be reframed "
                   "around d_int (or the latent geometry it indexes).")
    elif dint_spread < DINT_MOVES and oh_spread < OH_MOVES:
        verdict = ("D_HEAD-AS-SCALE SUPPORTED: with d_head fixed, plateau d_int AND O_h are "
                   "both ~flat. d_int moves only WITH d_head (between clusters), not within a "
                   "fixed-d_head family => d_head (or its associated scale) organizes O_h; "
                   "d_int is a downstream readout. (Peak d_int may still grow with size — "
                   "report peak vs plateau separately, per Valeriani.)")
    else:
        verdict = ("MIXED: d_int and O_h spreads disagree in direction/magnitude. Inspect the "
                   "profile — the peak (grows with size) and plateau (≈constant) may be telling "
                   "different stories; do not collapse them.")
    print(f"\n  => {verdict}")

    result = {"family": "gpt2", "d_head": GPT2_DHEAD, "rel_depth": rel_depth,
              "rows": rows, "spreads": {"d_int_at_rel": dint_spread, "O_h": oh_spread,
                                        "d_int_peak": peak_spread}, "verdict": verdict}
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n  wrote {out}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="ATLAS cheap experiment: d_int vs O_h with d_head fixed (GPT-2 family)")
    p.add_argument("--models", type=str, nargs="+",
                   default=["gpt2", "gpt2-medium", "gpt2-large"])
    p.add_argument("--n-blocks", type=int, default=10)
    p.add_argument("--n-points", type=int, default=1200)
    p.add_argument("--rel-depth", type=float, default=0.9)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()
    run(models=tuple(args.models), device=args.device, n_blocks=args.n_blocks,
        n_points=args.n_points, rel_depth=args.rel_depth, out=args.out)
