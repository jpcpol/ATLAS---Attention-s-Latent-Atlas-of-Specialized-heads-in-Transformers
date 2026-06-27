"""
Aggregate Batch-1 results — the §4 table + the P5 temporal-emergence curves.

Reads the per-run JSONs written by experiments/run_batch1.py and produces:
  (1) the final-state table: d_head -> O_h, plateau_d_int, VALID (only VALID rows count
      toward the §4 predictions);
  (2) the P5 emergence curves O_h(t), plateau_d_int(t) per run, and their discrete
      derivatives dO_h/dstep and d(d_int)/dstep, plus the ORDER of emergence — does the
      plateau-d_int reach its plateau before O_h organizes, or after?

No plotting here (that is plots.py); this is pure data reduction so it runs anywhere.
"""

from __future__ import annotations

import os
import sys
import json
import glob
import argparse


def load_runs(results_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


def _derivative(steps, vals):
    """Discrete dVal/dstep between consecutive snapshots (NaN-safe)."""
    out = []
    for i in range(1, len(steps)):
        ds = steps[i] - steps[i - 1]
        a, b = vals[i - 1], vals[i]
        if ds and a == a and b == b:
            out.append((steps[i], (b - a) / ds))
    return out


def _emergence_order(em):
    """Heuristic ordering: at which fraction does each curve reach 90% of its final move?
    Returns (frac_dint90, frac_oh90) or NaNs — descriptive, not a causal claim (P5)."""
    if len(em) < 2:
        return float("nan"), float("nan")
    fr = [e["frac"] for e in em]
    oh = [e["O_h"] for e in em]
    di = [e["plateau_d_int"] for e in em]

    def frac_at_90(vals):
        v0, vN = vals[0], vals[-1]
        span = vN - v0
        if abs(span) < 1e-9:
            return float("nan")
        target = v0 + 0.9 * span
        for f, v in zip(fr, vals):
            if (span > 0 and v >= target) or (span < 0 and v <= target):
                return f
        return fr[-1]

    return frac_at_90(di), frac_at_90(oh)


def summarize(runs):
    final_rows, emergence = [], []
    for r in runs:
        mc = r["config"]["model"]
        tag, dh, seed = r.get("config", {}).get("model", {}), mc["d_head"], r["seed"]
        em = r.get("emergence", [])
        steps = [e["step"] for e in em]
        oh = [e["O_h"] for e in em]
        di = [e["plateau_d_int"] for e in em]
        d_oh = _derivative(steps, oh)
        d_di = _derivative(steps, di)
        f_di90, f_oh90 = _emergence_order(em)
        final_rows.append({"d_head": dh, "seed": seed, "valid": r.get("VALID"),
                           "O_h": r.get("final_O_h"),
                           "plateau_d_int": r.get("final_plateau_d_int")})
        emergence.append({"d_head": dh, "seed": seed, "valid": r.get("VALID"),
                          "fracs": [e["frac"] for e in em], "O_h_t": oh, "d_int_t": di,
                          "dO_h_dstep": d_oh, "dd_int_dstep": d_di,
                          "frac_dint_90": f_di90, "frac_oh_90": f_oh90})
    return final_rows, emergence


def print_report(final_rows, emergence):
    print("=" * 72)
    print("[FINAL O_h TABLE]  (only VALID rows count toward §4 predictions)")
    print("=" * 72)
    print(f"  {'d_head':>6} | {'seed':>4} | {'valid':>5} | {'O_h':>6} | {'plateau_d_int':>13}")
    for r in sorted(final_rows, key=lambda x: (x["d_head"], x["seed"])):
        oh = f"{r['O_h']:.3f}" if r["O_h"] is not None else "  —  "
        di = f"{r['plateau_d_int']:.2f}" if r["plateau_d_int"] is not None else "  —  "
        print(f"  {r['d_head']:>6} | {r['seed']:>4} | {str(r['valid']):>5} | {oh:>6} | {di:>13}")

    print("\n" + "=" * 72)
    print("[P5 TEMPORAL EMERGENCE]  order: when does each curve reach 90% of its move?")
    print("=" * 72)
    print(f"  {'d_head':>6} | {'seed':>4} | {'frac@d_int90':>12} | {'frac@O_h90':>10} | who first?")
    for e in sorted(emergence, key=lambda x: (x["d_head"], x["seed"])):
        fd, fo = e["frac_dint_90"], e["frac_oh_90"]
        if fd == fd and fo == fo:
            who = "d_int" if fd < fo else ("O_h" if fo < fd else "together")
        else:
            who = "—"
        fds = f"{fd:.2f}" if fd == fd else " — "
        fos = f"{fo:.2f}" if fo == fo else " — "
        print(f"  {e['d_head']:>6} | {e['seed']:>4} | {fds:>12} | {fos:>10} | {who}")


def main():
    ap = argparse.ArgumentParser(description="Aggregate ablation Batch-1 results")
    ap.add_argument("--results-dir", type=str, default="../../../docs/ablation_batch1")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    runs = load_runs(args.results_dir)
    if not runs:
        print(f"no run JSONs found in {args.results_dir}"); return
    final_rows, emergence = summarize(runs)
    print_report(final_rows, emergence)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"final": final_rows, "emergence": emergence}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
