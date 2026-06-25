"""
NQP — Uncertainty Pareto frontier (EXP-U02, corrected formulation).

Tests conjecture NQP-U1b. The layer computes y = W a, and quantization injects
error from BOTH sides into the same output:

  ε_W = E_a ||(W - Ŵ) a||² / E_a||W a||²     (error from quantizing WEIGHTS)
  ε_A = E_a ||W (a - â)||² / E_a||W a||²       (error from quantizing ACTIVATIONS)

ε_W is minimized by quantizing W in the basis P_W that diagonalizes the
weight-side geometry; ε_A is minimized by quantizing a in the basis P_A that
diagonalizes G_A = E[a aᵀ]. EXP-U01 showed P_W and P_A do NOT commute (θ ≈ 49°).

The uncertainty arises under a SHARED basis constraint: if both weights and
activations must be expressed/quantized in one common basis U (as any real fused
kernel would prefer), then U cannot be optimal for both. We sweep a single shared
basis U(α) from P_A (α=0) to P_W (α=1) and quantize BOTH W and a in U(α):

  - α=0: activations quantize cleanly (ε_A low) but weights pay (ε_W high)
  - α=1: weights quantize cleanly (ε_W low) but activations pay (ε_A high)
  - non-commuting bases ⇒ no α kills both ⇒ a floor on ε_W·ε_A with an INTERIOR
    or trade-off structure, expected to grow with θ.

(The earlier version quantized only W and measured output error twice — that
collapses to "P_W is best" because F_W already absorbs G_A via the chain rule.
The fix: actually quantize activations too, so both sides are in play.)
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


def collect_activation_samples(
    model, dataloader, layer_names, n_batches=4, max_tokens=2048, device="cpu"
):
    """Capture a small buffer of input activations [N, d_in] per layer via hooks."""
    mods = {n: m for n, m in model.named_modules() if n in layer_names}
    buffers = {n: [] for n in mods}
    counts = {n: 0 for n in mods}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            if counts[name] >= max_tokens:
                return
            a = (inp[0] if isinstance(inp, tuple) else inp).reshape(-1, inp[0].shape[-1])
            take = min(a.shape[0], max_tokens - counts[name])
            buffers[name].append(a[:take].detach().to(device))
            counts[name] += take
        return hook

    for n, m in mods.items():
        handles.append(m.register_forward_hook(make_hook(n)))

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            model(input_ids=batch["input_ids"].to(device))

    for h in handles:
        h.remove()
    return {n: torch.cat(buffers[n], dim=0) for n in buffers if buffers[n]}


# ---------------------------------------------------------------------------
# Geodesic interpolation between two orthogonal bases
# ---------------------------------------------------------------------------

class BasisInterpolator:
    """
    Cheap orthogonal interpolation between two bases via linear blend + QR.

        U(α) = orthonormalize( (1-α)·P_A + α·P_W )

    This traces a continuous path of orthogonal bases from P_A (α=0) to P_W (α=1).
    It is NOT the exact SO(d) geodesic, but for tracing the ε_W/ε_A trade-off any
    smooth orthogonal path between the two optima suffices — and QR on 768×768 is
    ~100× cheaper than the complex eig the geodesic needs (which stalled EXP-U02).
    Endpoints are returned exactly.
    """
    def __init__(self, P_A: Tensor, P_W: Tensor):
        self.P_A = P_A
        self.P_W = P_W
        # Align column signs so the linear blend doesn't cancel near-parallel cols.
        signs = torch.sign((P_A * P_W).sum(dim=0)).clamp(min=-1.0)
        signs[signs == 0] = 1.0
        self.P_W_aligned = P_W * signs.unsqueeze(0)

    def at(self, alpha: float) -> Tensor:
        if alpha <= 0.0:
            return self.P_A
        if alpha >= 1.0:
            return self.P_W
        M = (1.0 - alpha) * self.P_A + alpha * self.P_W_aligned
        Q, R = torch.linalg.qr(M)
        return Q * torch.sign(torch.diag(R)).clamp(min=-1.0).unsqueeze(0)


def interpolate_basis(P_A: Tensor, P_W: Tensor, alpha: float) -> Tensor:
    """One-shot interpolation (for tests). For sweeps use BasisInterpolator."""
    return BasisInterpolator(P_A, P_W).at(alpha)


def _reorthonormalize(U: Tensor) -> Tensor:
    """QR re-orthonormalization safeguard (interpolation can drift numerically)."""
    Q, R = torch.linalg.qr(U)
    return Q * torch.sign(torch.diag(R)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Dual error measurement
# ---------------------------------------------------------------------------

def quantize_weight_in_basis(W: Tensor, U: Tensor, bits: int) -> Tensor:
    """Rotate W columns into basis U, per-column symmetric quant, rotate back. [d_out,d_in]"""
    U = _reorthonormalize(U)
    W_rot = W @ U
    n_levels = 2 ** (bits - 1) - 1
    scale = (W_rot.abs().amax(dim=0, keepdim=True) / n_levels).clamp(min=1e-9).expand_as(W_rot)
    return quantize_symmetric(W_rot, scale, bits) @ U.t()


def quantize_activations_in_basis(A: Tensor, U: Tensor, bits: int) -> Tensor:
    """
    Quantize activation samples A [N, d_in] in basis U, rotate back.
    Per-dimension scale set from the rotated activation range (static quant).
    """
    U = _reorthonormalize(U)
    A_rot = A @ U                                       # [N, d_in]
    n_levels = 2 ** (bits - 1) - 1
    scale = (A_rot.abs().amax(dim=0, keepdim=True) / n_levels).clamp(min=1e-9)
    A_q = quantize_symmetric(A_rot, scale.expand_as(A_rot), bits)
    return A_q @ U.t()


def dual_error_sampled(
    W: Tensor, W_hat: Tensor, A: Tensor, A_hat: Tensor
) -> tuple[float, float]:
    """
    Output-referred errors from quantizing weights vs activations, on real samples.

      ε_W = E_a||(W - Ŵ) a||² / E_a||W a||²       (weight quant error at the output)
      ε_A = E_a||W (a - â)||² / E_a||W a||²        (activation quant error at the output)

    A is [N, d_in]; outputs are A @ Wᵀ ∈ [N, d_out].
    """
    Y = A @ W.t()                                       # reference output [N, d_out]
    den = Y.pow(2).sum().item()
    dW = (A @ (W - W_hat).t()).pow(2).sum().item()      # ||(W-Ŵ)a||²
    dA = ((A - A_hat) @ W.t()).pow(2).sum().item()      # ||W(a-â)||²
    return dW / max(den, 1e-12), dA / max(den, 1e-12)


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
    items = list(fishers.items())
    if max_layers is not None:
        items = items[:max_layers]

    print(f"[EXP-U02] Collecting activation samples...")
    act = collect_activation_samples(
        model, loader, [n for n, _ in items], n_batches=4, device=device
    )

    alphas = [i / (n_alpha - 1) for i in range(n_alpha)]
    reports = []

    print(f"[EXP-U02] Tracing Pareto frontiers ({bits} bits, {n_alpha} α-steps)...\n")
    for name, (F_W, G_A) in items:
        if name not in mods or name not in act:
            continue
        W, _ = _weight_matrix(mods[name])              # [d_out, d_in]
        A = act[name]                                  # [N, d_in] real activations
        _, P_W = fisher_eig(F_W); P_W = P_W.flip(1)     # dominant first
        _, P_A = fisher_eig(G_A); P_A = P_A.flip(1)
        angle = math.degrees(principal_angles(P_W, P_A, min(top_k, P_W.shape[0])).mean().item())

        interp = BasisInterpolator(P_A, P_W)        # one eig per layer, reused for all α
        ew_list, ea_list = [], []
        for a in alphas:
            U = interp.at(a)
            W_hat = quantize_weight_in_basis(W, U, bits)
            A_hat = quantize_activations_in_basis(A, U, bits)
            ew, ea = dual_error_sampled(W, W_hat, A, A_hat)
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
