"""
NQP-Q — Geometric regularity of the residual manifold (EXP-Q05b / Q05c).

Q05 found dim_int(ε) ≈ 7 (LOCAL estimate). Before assuming M_ε is a smooth global
manifold (which an autoencoder Q06 would implicitly require), we test its global
geometry — the gap J.P. Chancay flagged: TwoNN gives local dim, not global regularity.

Q05b — SMOOTHNESS / connectivity:
  - kNN-graph connectivity: is M_ε one connected component or several patches?
  - Local-dim homogeneity: does TwoNN agree across random sub-regions (homogeneous)
    or vary wildly (patchwork / singularities)?
  - Linear-interpolation test: midpoints of pairs of residuals — do they stay near
    the data manifold (smooth/convex-ish) or fly off into empty space (curved/patchy)?

Q05c — INTERNAL STRUCTURE:
  - Do residuals cluster by layer / by sequence-position bucket? If the ~7D manifold
    is actually a union of per-layer or per-position submanifolds, the "single
    universal manifold" reading weakens.

All measured on residuals already collectible — forward-only, cheap.
"""

from __future__ import annotations

import math
import sys
import statistics

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from torch import Tensor

from intrinsic import twonn_dimension, collect_residuals


# ---------------------------------------------------------------------------
# Q05b — smoothness / connectivity
# ---------------------------------------------------------------------------

def knn_connectivity(X: Tensor, k: int = 10) -> tuple[int, float]:
    """
    Build a kNN graph and count connected components (union-find).
    Returns (n_components, frac_in_largest). 1 component & frac≈1 ⇒ connected manifold.
    """
    N = X.shape[0]
    d = torch.cdist(X, X)
    d.fill_diagonal_(float("inf"))
    knn = d.topk(k, dim=-1, largest=False).indices            # [N, k]

    parent = list(range(N))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    for i in range(N):
        for j in knn[i].tolist():
            union(i, j)
    roots = [find(i) for i in range(N)]
    from collections import Counter
    comp = Counter(roots)
    n_comp = len(comp)
    frac_largest = max(comp.values()) / N
    return n_comp, frac_largest


def local_dim_homogeneity(X: Tensor, n_regions: int = 6, region_size: int = 400):
    """
    Estimate TwoNN dim on several random sub-regions. Low spread ⇒ homogeneous
    manifold; high spread ⇒ patchwork / varying local dimension.
    """
    N = X.shape[0]
    dims = []
    for _ in range(n_regions):
        idx = torch.randperm(N)[:min(region_size, N)]
        dims.append(twonn_dimension(X[idx]))
    return statistics.mean(dims), statistics.pstdev(dims), dims


def interpolation_test(X: Tensor, n_pairs: int = 200, k: int = 5,
                       neighbor_frac: float = 0.25) -> float:
    """
    Curvature / convexity probe. For random NEARBY pairs (within the local neighborhood
    so the segment is short relative to manifold curvature), take the midpoint and
    measure its distance to the nearest data point, normalized by the LOCAL scale
    (median nearest-neighbor distance). On a locally-flat manifold a midpoint of two
    nearby points stays on it ⇒ ratio ~1. On a tightly curved manifold the chord cuts
    across empty space ⇒ ratio ≫1. Returns the median normalized off-manifold distance.
    """
    N = X.shape[0]
    d = torch.cdist(X, X)
    d.fill_diagonal_(float("inf"))
    nn1 = d.topk(1, dim=-1, largest=False).values.squeeze(-1)
    local_scale = nn1.median().clamp(min=1e-9)                # typical neighbor gap

    # pick pairs that are near each other (chord short vs curvature): bth percentile
    thresh = d.flatten().kthvalue(int(d.numel() * neighbor_frac)).values
    ia = torch.randperm(N)[:n_pairs]
    offs = []
    for i in ia.tolist():
        cand = (d[i] < thresh).nonzero(as_tuple=True)[0]
        if len(cand) == 0:
            continue
        j = cand[torch.randint(len(cand), (1,)).item()].item()
        mid = 0.5 * (X[i] + X[j])
        nearest = torch.cdist(mid.unsqueeze(0), X).min().item()
        offs.append(nearest / local_scale.item())
    return float(torch.tensor(offs).median()) if offs else float("nan")


# ---------------------------------------------------------------------------
# Q05c — internal structure (clustering by layer / position)
# ---------------------------------------------------------------------------

def submanifold_separation(res_by_layer: dict) -> float:
    """
    Do residuals separate by layer? Compare mean intra-layer distance to mean
    inter-layer distance (a silhouette-like ratio). ratio≈1 ⇒ no separation (one
    manifold); ratio≪1 ⇒ layers form distinct submanifolds.
    """
    layers = sorted(res_by_layer.keys())
    cents = {l: res_by_layer[l].mean(0) for l in layers}
    intra = statistics.mean(
        (res_by_layer[l] - cents[l]).norm(dim=-1).mean().item() for l in layers
    )
    inter_vals = []
    for i, li in enumerate(layers):
        for lj in layers[i+1:]:
            inter_vals.append((cents[li] - cents[lj]).norm().item())
    inter = statistics.mean(inter_vals) if inter_vals else 1.0
    return intra / max(inter, 1e-9)


def atlas_test(res_by_head: dict, d_local: int = 7):
    """
    Q05d — is the head-wise structure a genuine ATLAS (incompatible coordinates) or
    just OFFSET copies of one geometry (same basis, different center)?

    Two discriminating measures:
      1. subspace_overlap: mean principal-angle alignment between the top-d_local PCA
         bases of head pairs. ~1 ⇒ shared coordinates (NOT an atlas); ~0 ⇒ orthogonal
         coordinates (genuine atlas).
      2. centered_pooled_dim: intrinsic dim of the pooled residuals AFTER centering
         each head at its own mean. If it collapses to ≈d_local ⇒ mere offsets; if it
         stays high (≈pooled) ⇒ genuinely incompatible coordinates (atlas).
    """
    keys = sorted(res_by_head.keys())
    # per-head local PCA basis (top d_local directions) via SVD (robust to degeneracy)
    bases = {}
    for k in keys:
        E = res_by_head[k]
        Ec = E - E.mean(0, keepdim=True)
        # right singular vectors of centered data = principal directions
        _, _, Vt = torch.linalg.svd(Ec, full_matrices=False)
        bases[k] = Vt[:d_local].t()                       # [dh, d_local]

    # 1. mean pairwise subspace overlap (mean cos of principal angles)
    overlaps = []
    for i, ki in enumerate(keys):
        for kj in keys[i+1:]:
            s = torch.linalg.svdvals(bases[ki].t() @ bases[kj]).clamp(-1, 1)
            overlaps.append(s.mean().item())             # mean cosine = alignment
    mean_overlap = sum(overlaps) / max(len(overlaps), 1)

    # 2. intrinsic dim of head-centered pooled residuals
    centered = torch.cat([res_by_head[k] - res_by_head[k].mean(0, keepdim=True)
                          for k in keys], 0)
    if centered.shape[0] > 2500:
        centered = centered[torch.randperm(centered.shape[0])[:2500]]
    centered_dim = twonn_dimension(centered)

    return mean_overlap, centered_dim


def run_q05bc(layers=(9, 10, 11), n_blocks=8, device="cpu", seed=42):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset
    import os
    torch.manual_seed(seed)
    if str(device) == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    print("[Q05b/c] Loading GPT-2 small...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2"); model.eval()
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)

    res = collect_residuals(model, ids, 256, device, layers, n_blocks=n_blocks, max_points=2000)
    # keep per-head (Q05d) and pooled-per-layer (Q05c)
    by_head = {k: E for k, E in res.items()}
    by_layer = {}
    for (l, h), E in res.items():
        by_layer.setdefault(l, []).append(E)
    by_layer = {l: torch.cat(v, 0) for l, v in by_layer.items()}
    pooled = torch.cat(list(by_layer.values()), 0)
    if pooled.shape[0] > 2500:
        pooled = pooled[torch.randperm(pooled.shape[0])[:2500]]

    print(f"\n{'='*66}\n[Q05b] Smoothness / connectivity of M_ε (N={pooled.shape[0]})\n{'='*66}")
    n_comp, frac = knn_connectivity(pooled, k=10)
    mdim, sdim, _ = local_dim_homogeneity(pooled)
    off = interpolation_test(pooled)
    print(f"  kNN connectivity: {n_comp} component(s), {frac:.1%} in largest")
    print(f"  local-dim homogeneity: {mdim:.1f} ± {sdim:.1f}  (low std ⇒ homogeneous)")
    print(f"  interpolation off-manifold (median, norm.): {off:.2f}  "
          f"(≈1 ⇒ midpoints leave manifold / curved; <1 ⇒ locally flat)")

    print(f"\n{'='*66}\n[Q05c] Internal structure (per-layer submanifolds)\n{'='*66}")
    ratio = submanifold_separation(by_layer)
    print(f"  intra/inter-layer distance ratio: {ratio:.2f}  "
          f"(≈1 ⇒ single manifold; ≪1 ⇒ distinct per-layer submanifolds)")

    print(f"\n{'='*66}\n[Q05d] Atlas test — shared coords vs head-specific coords\n{'='*66}")
    overlap, centered_dim = atlas_test(by_head, d_local=7)
    print(f"  mean head-pair subspace overlap (cos principal angles): {overlap:.3f}  "
          f"(~1 ⇒ shared basis; ~0 ⇒ orthogonal/atlas)")
    print(f"  head-CENTERED pooled intrinsic dim: {centered_dim:.1f}  "
          f"(≈7 ⇒ mere offsets; ≈pooled ⇒ incompatible coords/atlas)")

    print(f"\n{'='*66}\n[Q05b/c/d VERDICT]\n{'='*66}")
    connected = (n_comp == 1 or frac > 0.95)
    homogeneous = sdim < 0.25 * mdim
    single = ratio > 0.5
    print(f"  connected manifold:   {'YES' if connected else 'NO'} ({n_comp} comp, {frac:.0%})")
    print(f"  homogeneous local dim:{'YES' if homogeneous else 'NO'} (std {sdim:.1f}/mean {mdim:.1f})")
    print(f"  single (not per-layer split): {'YES' if single else 'NO'} (ratio {ratio:.2f})")
    # Atlas decision driven by SUBSPACE OVERLAP (validated discriminator: atlas→0,
    # shared→1). centered_dim is a secondary, noisier cross-check.
    if overlap < 0.4:
        print(f"  => ATLAS STRUCTURE: heads use INCOMPATIBLE coordinates "
              f"(overlap {overlap:.2f} ≈ orthogonal). NOT a single global manifold — "
              f"fiber-bundle-like. Q06 must be PER-HEAD or routing/atlas-learning, "
              f"NOT a global autoencoder.")
        atlas = True
    elif overlap > 0.75:
        print(f"  => SHARED COORDINATES: heads align strongly (overlap {overlap:.2f}). "
              f"A (near-)global basis exists; global AE viable. NOT an atlas.")
        atlas = False
    else:
        print(f"  => PARTIAL ATLAS: heads share coordinates partially (overlap {overlap:.2f}). "
              f"Mixed structure; per-head AE safest, shared basis captures only part.")
        atlas = (overlap < 0.5)
    return {"n_comp": n_comp, "frac_largest": frac, "local_dim_mean": mdim,
            "local_dim_std": sdim, "interp_off": off, "layer_ratio": ratio,
            "head_overlap": overlap, "centered_dim": centered_dim, "atlas": atlas}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="NQP-Q EXP-Q05b/c — residual manifold geometry")
    p.add_argument("--layers", type=int, nargs="+", default=[9, 10, 11])
    p.add_argument("--n-blocks", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    run_q05bc(layers=tuple(args.layers), n_blocks=args.n_blocks, device=args.device)
