"""
Plots for Batch-1 — the §4 O_h(d_head) curve and the P5 temporal-emergence curves.

Two figures:
  (1) O_h vs d_head (final, VALID runs only) — the P1 test, with the cross-arch clusters
      (GPT-2/Qwen ~0.28, Llama/Mistral ~0.20) as reference lines.
  (2) Temporal emergence: O_h(t) and plateau-d_int(t) over training fraction, per run —
      the P5 picture of what forms first.

matplotlib only (as the rest of the project). Reads the aggregate JSON or the raw run dir.
"""

from __future__ import annotations

import os
import json
import glob
import argparse


def _load(results_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if "emergence" in r and "config" in r:          # a raw run file
            runs.append(r)
    return runs


def plot_oh_vs_dhead(runs, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [(r["config"]["model"]["d_head"], r["final_O_h"])
           for r in runs if r.get("VALID") and r.get("final_O_h") is not None]
    if not pts:
        print("  (no VALID runs with O_h yet — skipping O_h-vs-d_head plot)")
        return
    pts.sort()
    xs, ys = zip(*pts)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0.28, ls="--", c="tab:blue", alpha=0.5, label="cross-arch d_head 64 (~0.28)")
    ax.axhline(0.20, ls="--", c="tab:red", alpha=0.5, label="cross-arch d_head 128 (~0.20)")
    ax.scatter(xs, ys, c="k", zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("d_head"); ax.set_ylabel("inter-head O_h (k=7)")
    ax.set_title("P1: O_h vs d_head (VALID ablation runs)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"  wrote {out_path}")


def plot_emergence(runs, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    runs = [r for r in runs if r.get("emergence")]
    if not runs:
        print("  (no emergence data — skipping emergence plot)")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for r in runs:
        em = r["emergence"]
        dh = r["config"]["model"]["d_head"]; seed = r["seed"]
        fr = [e["frac"] for e in em]
        lbl = f"d_head={dh} s{seed}"
        ax1.plot(fr, [e["O_h"] for e in em], marker="o", label=lbl)
        ax2.plot(fr, [e["plateau_d_int"] for e in em], marker="o", label=lbl)
    ax1.set_xlabel("training fraction"); ax1.set_ylabel("O_h"); ax1.set_title("O_h(t)")
    ax2.set_xlabel("training fraction"); ax2.set_ylabel("plateau d_int")
    ax2.set_title("plateau d_int(t)")
    ax1.legend(fontsize=7); ax2.legend(fontsize=7)
    fig.suptitle("P5: temporal emergence of the atlas")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot ablation Batch-1 results")
    ap.add_argument("--results-dir", type=str, default="../../../docs/ablation_batch1")
    ap.add_argument("--out-dir", type=str, default="../../../docs/figures")
    args = ap.parse_args()
    runs = _load(args.results_dir)
    if not runs:
        print(f"no run JSONs in {args.results_dir}"); return
    os.makedirs(args.out_dir, exist_ok=True)
    plot_oh_vs_dhead(runs, os.path.join(args.out_dir, "ablation_oh_vs_dhead.png"))
    plot_emergence(runs, os.path.join(args.out_dir, "ablation_emergence.png"))


if __name__ == "__main__":
    main()
