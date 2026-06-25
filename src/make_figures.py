"""
NQP-Q — Render all paper figures from docs/figure_data.json into docs/figures/.

Main:
  fig1_overlap_matrix    — H×H overlap heatmaps (small/medium/large). ICONIC.
  fig2_overlap_vs_scale  — O_h vs model scale with bootstrap CIs.
  fig3_pca_vs_intrinsic  — linear (PCA ~30D) vs intrinsic (TwoNN ~7D) per head.
  fig4_pipeline          — conceptual narrative pipeline.
  fig5_atlas_schematic   — pedagogical atlas illustration (NOT evidence).
Supplementary:
  figS1_dlocal_sweep     — O_h(k) for k=4..10 (neutralizes d_local critique).
  figS2_intercorpus      — O_h WikiText vs C4 with CIs.
  figS3_q06_negative     — PCA rank-7 vs AE 64→7→64 recovery barplot.
"""

from __future__ import annotations

import os
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Ellipse

OUT = "../docs/figures"
DATA = "../docs/figure_data.json"

MODELS = ["gpt2", "gpt2-medium", "gpt2-large"]
NICE = {"gpt2": "GPT-2 small (124M)", "gpt2-medium": "GPT-2 medium (355M)",
        "gpt2-large": "GPT-2 large (774M)"}


def load():
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
def fig1_overlap_matrix(d):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, m in zip(axes, MODELS):
        M = np.array(d["models"][m]["overlap_matrix"])
        labels = d["models"][m]["labels"]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
        off = M[~np.eye(M.shape[0], dtype=bool)].mean()
        ax.set_title(f"{NICE[m]}\nmean off-diag $O_h$ = {off:.3f}", fontsize=10)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.6 else "black", fontsize=6)
    fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02, label="subspace overlap")
    fig.suptitle("Inter-head residual subspace overlap (diagonal = 1, off-diagonal ≈ 0.28): "
                 "heads are geometrically decoupled modules", fontsize=11, y=1.02)
    _save(fig, "fig1_overlap_matrix")


def fig2_overlap_vs_scale(d):
    ci = d["overlap_ci"]
    xs = [ci[m]["params_M"] for m in MODELS]
    ys = [ci[m]["O_h"] for m in MODELS]
    lo = [ci[m]["O_h"] - ci[m]["lo"] for m in MODELS]
    hi = [ci[m]["hi"] - ci[m]["O_h"] for m in MODELS]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", capsize=5, color="#2c3e80",
                markersize=8, lw=2, label="$O_h$ (95% bootstrap CI)")
    ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.text(xs[-1], 0.97, "shared-subspace expectation ($O_h=1$)", ha="right",
            va="top", fontsize=8, color="gray")
    ax.set_xscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([f"{x}M" for x in xs])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("model size (parameters, log scale)")
    ax.set_ylabel("inter-head overlap  $O_h$")
    ax.set_title("Non-alignment is invariant across a 6× change in scale\n"
                 "(CIs overlap; $O_h \\ll 1$ throughout)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, "fig2_overlap_vs_scale")


def fig3_pca_vs_intrinsic(d):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(MODELS)); w = 0.35
    lin = [np.mean(d["models"][m]["linear_dims"]) for m in MODELS]
    int = [np.mean(d["models"][m]["intrinsic_dims"]) for m in MODELS]
    lin_e = [np.std(d["models"][m]["linear_dims"]) for m in MODELS]
    int_e = [np.std(d["models"][m]["intrinsic_dims"]) for m in MODELS]
    ax.bar(x - w/2, lin, w, yerr=lin_e, capsize=4, label="linear rank (PCA, 90% var)",
           color="#b0b8c8")
    ax.bar(x + w/2, int, w, yerr=int_e, capsize=4, label="intrinsic dim (TwoNN)",
           color="#2c3e80")
    for xi, v in zip(x - w/2, lin): ax.text(xi, v + 0.6, f"{v:.0f}", ha="center", fontsize=8)
    for xi, v in zip(x + w/2, int): ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([NICE[m].split(" (")[0] for m in MODELS], fontsize=8)
    ax.set_ylabel("dimension of residual ε")
    ax.set_title("The residual is nonlinear: intrinsic dim (~7) ≪ linear rank (~30)\n"
                 "— why linear compression (Q03) misses the structure", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig3_pca_vs_intrinsic")


def fig4_pipeline(_):
    steps = [
        ("Top-$k$ selection fails", "+7.7 PPL: tail does\nsystematic integration", "#e74c3c"),
        ("Low-rank fails", "ε needs ~30 linear dims\n(PCA, 90% var)", "#e74c3c"),
        ("Intrinsic manifold", "TwoNN ≈ 7 ≪ 30:\nnonlinear, low-dim", "#27ae60"),
        ("Atlas", "heads non-aligned\n$O_h$ ≈ 0.28, scale-invariant", "#27ae60"),
        ("AE negative", "per-head 64→7→64 ≈ PCA:\ngeometric, not compressible", "#e67e22"),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 8.6)); ax.axis("off")
    y = 0.92; dy = 0.185
    for i, (title, sub, c) in enumerate(steps):
        box = FancyBboxPatch((0.12, y - 0.07), 0.76, 0.12,
                             boxstyle="round,pad=0.02", fc=c, ec="black", alpha=0.85)
        ax.add_patch(box)
        ax.text(0.5, y + 0.005, title, ha="center", va="center", fontsize=11,
                color="white", weight="bold")
        ax.text(0.5, y - 0.045, sub, ha="center", va="center", fontsize=7.5, color="white")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((0.5, y - 0.075), (0.5, y - dy + 0.055),
                         arrowstyle="-|>", mutation_scale=18, color="black", lw=1.5))
        y -= dy
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Narrative: ruling out the obvious, then characterizing\nwhat remains",
                 fontsize=11)
    _save(fig, "fig4_pipeline")


def fig5_atlas_schematic(_):
    fig, ax = plt.subplots(figsize=(6.5, 4.5)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    # ambient box
    ax.add_patch(plt.Rectangle((0.4, 0.4), 9.2, 6.2, fill=False, ec="gray", ls="--"))
    ax.text(9.4, 0.65, r"$\mathbb{R}^{64}$ (ambient)", ha="right", fontsize=10, color="gray")
    # three non-aligned local charts (ellipses at different orientations)
    charts = [((2.5, 4.6), 35), ((5.2, 2.3), -20), ((7.6, 4.9), 80)]
    for i, ((cx, cy), ang) in enumerate(charts, 1):
        ax.add_patch(Ellipse((cx, cy), 2.6, 1.0, angle=ang, fc="#2c3e80",
                     ec="black", alpha=0.55))
        ax.text(cx, cy, f"$M_{i}$", ha="center", va="center", color="white",
                fontsize=13, weight="bold")
        ax.text(cx, cy - 0.95, f"head {i}\n~7D", ha="center", va="top", fontsize=7.5)
    ax.text(5.0, 6.25, "Each head's residual = a low-dim chart with its OWN coordinates",
            ha="center", fontsize=9.5)
    ax.text(5.0, 0.05, "Charts are mutually non-aligned ($O_h$ ≈ 0.28), not a shared manifold",
            ha="center", fontsize=9.5, color="#2c3e80")
    ax.set_title("Conceptual illustration of the head-specific manifold atlas\n"
                 "(pedagogical — not empirical evidence)", fontsize=10)
    _save(fig, "fig5_atlas_schematic")


def figS1_dlocal_sweep(d):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for m in MODELS:
        sw = d["models"][m]["dlocal_sweep"]
        ks = sorted(int(k) for k in sw)
        ax.plot(ks, [sw[str(k)] for k in ks], "o-", label=NICE[m].split(" (")[0])
    ax.axvline(7, ls=":", color="gray"); ax.text(7.1, 0.15, "reported $d_{local}=7$", fontsize=8)
    ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.set_xlabel("$d_{local}$ (frame dimension)"); ax.set_ylabel("$O_h$")
    ax.set_ylim(0, 1.05)
    ax.set_title("$O_h(d_{local})$: the value scales with $d_{local}$,\n"
                 "but $O_h \\ll 1$ for every choice — conclusion is robust", fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    _save(fig, "figS1_dlocal_sweep")


def figS2_intercorpus(d):
    ic = d["intercorpus"]
    names = ["WikiText-103", "C4"]; keys = ["wikitext", "c4"]
    ys = [ic[k]["O_h"] for k in keys]
    lo = [ic[k]["O_h"] - ic[k]["lo"] for k in keys]
    hi = [ic[k]["hi"] - ic[k]["O_h"] for k in keys]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.bar(names, ys, yerr=[lo, hi], capsize=6, color=["#2c3e80", "#3a7ca5"], width=0.5)
    for i, v in enumerate(ys): ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.0); ax.axhline(1.0, ls="--", color="gray", lw=1)
    ax.set_ylabel("$O_h$ (GPT-2 small, deepest layer)")
    ax.set_title("Inter-corpus control: non-alignment is\ndata-independent (CIs overlap)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "figS2_intercorpus")


def figS3_q06_negative(d):
    q = d["q06"]
    fig, ax = plt.subplots(figsize=(5, 4.2))
    names = ["PCA rank-7\n(linear)", "AE 64→7→64\n(nonlinear)"]
    ys = [q["pca_rank7_recovered"], q["ae_64_7_64_recovered"]]
    ax.bar(names, ys, color=["#b0b8c8", "#2c3e80"], width=0.55)
    for i, v in enumerate(ys): ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of Top-1 PPL damage recovered")
    ax.set_title("Q06 negative: a nonlinear per-head autoencoder\n"
                 "matches but does NOT beat linear PCA at $d=7$", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "figS3_q06_negative")


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png")


def main():
    d = load()
    print("[make_figures] rendering...")
    fig1_overlap_matrix(d)
    fig2_overlap_vs_scale(d)
    fig3_pca_vs_intrinsic(d)
    fig4_pipeline(d)
    fig5_atlas_schematic(d)
    figS1_dlocal_sweep(d)
    figS2_intercorpus(d)
    figS3_q06_negative(d)
    print(f"[make_figures] done → {OUT}/")


if __name__ == "__main__":
    main()
