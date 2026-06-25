"""
NQP — Uncertainty Pareto frontier (EXP-U02).

Tests conjecture NQP-U1b: if the weight-Fisher basis P_W and activation-Fisher
basis P_A do not commute (shown in EXP-U01, angle ≈ 49°), then weight-precision
ε_W and activation-precision ε_A cannot be minimized simultaneously — there is a
floor on the product ε_W · ε_A, and that floor should grow with the angle θ
between the bases (theory/uncertainty_principle.md §4, NQP-U1b).

Method (per linear layer y = W a):
  - Build an interpolated quantization basis U(α) that rotates from P_A (α=0,
    optimal for activations) to P_W (α=1, optimal for weights) along the geodesic.
  - Quantize W in basis U(α) at fixed bits; measure BOTH:
      ε_W = ||W - Ŵ||² / ||W||²                     (weight reconstruction)
      ε_A = E_a ||W a - Ŵ a||² / E_a||W a||²         (output / activation error)
  - Sweep α ∈ [0,1] → trace the (ε_W, ε_A) frontier.
  - If bases commuted, one α would minimize both (frontier collapses to a point /
    L-shape touching the origin). Non-commuting ⇒ convex trade-off with a floor.
  - Correlate the floor min_α(ε_W·ε_A) with the per-layer angle θ from EXP-U01.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch import Tensor

from fisher import quantize_symmetric, quant_error_l2
from fisher_block import _linear_modules, _weight_matrix, fisher_eig
from uncertainty import estimate_both_fishers, principal_angles


# ---------------------------------------------------------------------------
# Geodesic interpolation between two orthogonal bases
# ---------------------------------------------------------------------------

def interpolate_basis(P_A: Tensor, P_W: Tensor, alpha: float) -> Tensor:
    """
    Orthogonal basis on the geodesic from P_A (alpha=0) to P_W (alpha=1).

    Compute the relative rotation R = P_Aᵀ P_W, take its matrix log to get the
    generator, scale by alpha, exponentiate, and apply. Result stays orthogonal.

        U(α) = P_A · expm( α · logm(P_Aᵀ P_W) )

    For numerical robustness we use an SVD-based orthogonal interpolation: rotating
    via the principal-angle decomposition keeps U(α) exactly orthogonal at every α.
    """
    if alpha <= 0.0:
        return P_A
    if alpha >= 1.0:
        return P_W
    R = P_A.t() @ P_W                       # relative rotation [d,d]
    # Orthogonal Procrustes / SVD: R = X S Yᵀ ; nearest rotation interpolation
    X, S, Yt = torch.linalg.svd(R)
    # Geodesic on SO(d): U(α) = P_A X · diag(angle·α via S) · Yᵀ ... approximate with
    # slerp of the orthogonal factor (S≈1 since R is near-orthogonal for close bases).
    Q = X @ Yt                              # orthogonal part of R (nearest rotation)
    # Matrix-log of Q (skew-symmetric generator), scale, re-exponentiate.
    # eigendecomp of skew via real Schur is overkill here; use a stable series:
    A = _matrix_log_orthogonal(Q)
    Qa = torch.linalg.matrix_exp(alpha * A)
    return P_A @ Qa


def _matrix_log_orthogonal(Q: Tensor) -> Tensor:
    """Skew-symmetric log of an orthogonal matrix via eigendecomposition."""
    # Q orthogonal ⇒ complex eigenvalues on unit circle. Use the real form:
    # log(Q) is skew-symmetric; approximate stably with (Q - Qᵀ)/2 projected.
    # For small rotations this is accurate; for large, matrix_exp(α·log) still
    # yields an orthogonal interpolant (we re-orthonormalize as a safeguard).
    S = 0.5 * (Q - Q.t())                   # skew part = first-order log
    return S


def _reorthonormalize(U: Tensor) -> Tensor:
    """QR re-orthonormalization safeguard (interpolation can drift numerically)."""
    Q, R = torch.linalg.qr(U)
    return Q * torch.sign(torch.diag(R)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Dual error measurement
# ---------------------------------------------------------------------------

def quantize_in_basis(W: Tensor, U: Tensor, bits: int) -> Tensor:
    """Rotate W into basis U, per-column symmetric quant, rotate back."""
    U = _reorthonormalize(U)
    W_rot = W @ U
    n_levels = 2 ** (bits - 1) - 1
    scale = (W_rot.abs().amax(dim=0, keepdim=True) / n_levels).clamp(min=1e-9).expand_as(W_rot)
    return quantize_symmetric(W_rot, scale, bits) @ U.t()


def dual_error(W: Tensor, W_hat: Tensor, A_cov: Tensor) -> tuple[float, float]:
    """
    Return (ε_W, ε_A):
      ε_W = ||W - Ŵ||² / ||W||²                          (weight space)
      ε_A = tr(ΔWᵀ ΔW · G_A) / tr(Wᵀ W · G_A)            (output space, weighted by
            activation covariance G_A = E[a aᵀ]; this is E_a||ΔW a||²/E_a||W a||²).
    """
    dW = W - W_hat
    eps_w = dW.pow(2).sum().item() / max(W.pow(2).sum().item(), 1e-12)
    # E_a||M a||² = tr(M G_A Mᵀ) = sum over rows of (M G_A) ⊙ M
    num = (dW @ A_cov * dW).sum().item()
    den = (W @ A_cov * W).sum().item()
    eps_a = num / max(den, 1e-12)
    return eps_w, eps_a


# ---------------------------------------------------------------------------
# EXP-U02
# ---------------------------------------------------------------------------

@dataclass
class ParetoReport:
    name: str
    angle_deg: float
    alphas: list
    eps_w: list
    eps_a: list
    floor_product: float        # min_α (ε_W · ε_A)
    eps_w_at_floor: float
    eps_a_at_floor: float

    def __repr__(self) -> str:
        return (
            f"{self.name:34s}  θ={self.angle_deg:5.1f}°  "
            f"floor(εW·εA)={self.floor_product:.3e}  "
            f"@(εW={self.eps_w_at_floor:.3e}, εA={self.eps_a_at_floor:.3e})"
        )


def run_exp_u02(
    n_calib: int = 64,
    bits: int = 4,
    top_k: int = 16,
    n_alpha: int = 9,
    device: str = "cpu",
    seed: int = 42,
    max_layers: int | None = None,
) -> dict:
    """
    EXP-U02 — Trace the ε_W/ε_A Pareto frontier per layer and test NQP-U1b:
    does the product floor correlate with the basis angle θ?
    """
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        from datasets import load_dataset
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise ImportError("requires transformers, datasets, torch") from e

    torch.manual_seed(seed)
    import os
    if str(device) == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    print("[EXP-U02] Loading GPT-2 small...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    print("[EXP-U02] Loading WikiText-103 calibration...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    subset = ds["train"].select(range(min(n_calib * 8, len(ds["train"]))))

    def tok_fn(ex):
        enc = tok(ex["text"], truncation=True, max_length=512, padding="max_length")
        enc["labels"] = [
            [t if m == 1 else -100 for t, m in zip(ids, msk)]
            for ids, msk in zip(enc["input_ids"], enc["attention_mask"])
        ]
        return enc

    tokenized = subset.map(tok_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    loader = DataLoader(tokenized, batch_size=4, shuffle=False)

    print(f"[EXP-U02] Estimating F_W and G_A ({n_calib} samples)...")
    fishers = estimate_both_fishers(model, loader, n_samples=n_calib, device=device)

    mods = _linear_modules(model)
    alphas = [i / (n_alpha - 1) for i in range(n_alpha)]
    reports = []

    print(f"[EXP-U02] Tracing Pareto frontiers ({bits} bits, {n_alpha} α-steps)...\n")
    items = list(fishers.items())
    if max_layers is not None:
        items = items[:max_layers]

    for name, (F_W, G_A) in items:
        if name not in mods:
            continue
        W, _ = _weight_matrix(mods[name])              # [d_out, d_in]
        _, P_W = fisher_eig(F_W); P_W = P_W.flip(1)     # dominant first
        _, P_A = fisher_eig(G_A); P_A = P_A.flip(1)
        angle = math.degrees(principal_angles(P_W, P_A, min(top_k, P_W.shape[0])).mean().item())

        ew_list, ea_list = [], []
        for a in alphas:
            U = interpolate_basis(P_A, P_W, a)
            W_hat = quantize_in_basis(W, U, bits)
            ew, ea = dual_error(W, W_hat, G_A)
            ew_list.append(ew)
            ea_list.append(ea)

        products = [w * a for w, a in zip(ew_list, ea_list)]
        i_floor = min(range(len(products)), key=lambda i: products[i])
        rep = ParetoReport(
            name=name, angle_deg=angle, alphas=alphas,
            eps_w=ew_list, eps_a=ea_list,
            floor_product=products[i_floor],
            eps_w_at_floor=ew_list[i_floor],
            eps_a_at_floor=ea_list[i_floor],
        )
        reports.append(rep)
        print(" ", rep)

    # ── Test NQP-U1b: does the floor correlate with the angle? ───────────────
    angles = [r.angle_deg for r in reports]
    floors = [math.log10(max(r.floor_product, 1e-30)) for r in reports]
    corr = _pearson(angles, floors)

    # Also: how often is the joint optimum NOT at an endpoint (genuine interior
    # trade-off ⇒ the bases really conflict, can't satisfy both at α=0 or α=1)?
    interior = sum(
        1 for r in reports
        if 0 < min(range(len(r.eps_w)), key=lambda i: r.eps_w[i] * r.eps_a[i]) < len(r.alphas) - 1
    )

    print(f"\n{'='*72}\n[EXP-U02 VERDICT]  test NQP-U1b (floor ε_W·ε_A vs angle θ)\n{'='*72}")
    print(f"  layers analyzed:                 {len(reports)}")
    print(f"  interior joint-optimum (true trade-off): {interior}/{len(reports)}")
    print(f"  Pearson corr( θ , log floor ):   {corr:+.3f}")
    if corr > 0.3 and interior > len(reports) // 2:
        print(f"  => NQP-U1b SUPPORTED: larger basis angle ⇒ higher ε_W·ε_A floor, and the "
              f"joint optimum is interior in most layers — a genuine uncertainty trade-off.")
    elif interior <= len(reports) // 2:
        print(f"  => NQP-U1b WEAK: joint optimum usually at an endpoint — one basis serves "
              f"both, little trade-off despite non-commuting bases.")
    else:
        print(f"  => NQP-U1b PARTIAL: trade-off exists but floor does not track θ "
              f"(corr={corr:+.2f}); the bound may depend on spectra, not angle alone.")

    return {"reports": reports, "corr_angle_floor": corr, "interior": interior,
            "n_layers": len(reports)}


def _pearson(x: list, y: list) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / max(sx * sy, 1e-12)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="NQP EXP-U02 — uncertainty Pareto frontier")
    p.add_argument("--n-calib", type=int, default=64)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--n-alpha", type=int, default=9)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--max-layers", type=int, default=None)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    run_exp_u02(n_calib=args.n_calib, bits=args.bits, n_alpha=args.n_alpha,
                top_k=args.top_k, max_layers=args.max_layers, device=args.device)
