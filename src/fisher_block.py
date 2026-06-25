"""
NQP — Block-Fisher quantization with real rotation (Camino A).

This is the first instantiation of P̂ = U ≠ I — the actual "preparation operator"
that rotates weights into the eigenbasis of the per-layer Fisher before quantizing.
The diagonal version (src/fisher.py) is the degenerate U = I ablation.

Key idea (operator_formalization.md §3):
  For a weight matrix W ∈ R^{d_out × d_in}, the input-side Fisher
      F = E_x[ a aᵀ ]   with a = layer input activations
  captures which input directions matter for the loss. Diagonalize F = U Λ Uᵀ,
  rotate W → W̃ = W U (so columns of W̃ are in the Fisher eigenbasis), quantize
  per-column with Fisher-aware scale, then reconstruct Ŵ = Q(W̃) Uᵀ.

  This is the SAME F that GPTQ uses (Hessian of layer output ≈ input second moment),
  but instead of GPTQ's error-feedback we do an explicit basis rotation — which lets
  us run the decisive ablation:

GATE A-G4 (the central scientific question):
  Does rotating into the FISHER eigenbasis beat rotating into a RANDOM orthogonal
  basis (QuIP-style) at the same bit budget? If not, NQP collapses to QuIP and the
  Fisher structure adds nothing. This module runs that comparison first.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Iterable

# Windows consoles default to cp1252; force UTF-8 so prints with math symbols work.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch import Tensor


# Reuse primitives from the diagonal module
from fisher import quantize_symmetric, quant_error_l2, _should_quantize


# ---------------------------------------------------------------------------
# Input-activation (block) Fisher via forward hooks
# ---------------------------------------------------------------------------

@dataclass
class BlockFisher:
    """Per-linear-layer input second-moment matrices F = E[a aᵀ]."""
    cov: dict[str, Tensor] = field(default_factory=dict)   # module_name → [d_in, d_in]
    count: dict[str, int] = field(default_factory=dict)

    def finalize(self) -> dict[str, Tensor]:
        """Return averaged covariance per layer."""
        return {k: self.cov[k] / max(self.count[k], 1) for k in self.cov}


def _linear_modules(model: nn.Module) -> dict[str, nn.Module]:
    """
    Collect modules whose weights NQP quantizes. Covers nn.Linear and HF Conv1D
    (GPT-2 uses Conv1D for attn/mlp projections: weight shape [in, out]).
    """
    mods = {}
    for name, m in model.named_modules():
        cls = m.__class__.__name__
        if cls in ("Linear", "Conv1D") and hasattr(m, "weight"):
            wname = f"{name}.weight"
            if _should_quantize(wname, m.weight.data):
                mods[name] = m
    return mods


def estimate_block_fisher(
    model: nn.Module,
    dataloader: Iterable,
    n_samples: int = 128,
    device: str | torch.device = "cpu",
) -> dict[str, Tensor]:
    """
    Estimate input-activation covariance F = E[a aᵀ] for each linear layer via hooks.

    This is the Gauss-Newton / input-second-moment approximation of the Fisher —
    the same object GPTQ inverts. We collect it to diagonalize and rotate.

    Returns: dict module_name → F ∈ R^{d_in × d_in}.
    """
    model.eval()
    model.to(device)
    mods = _linear_modules(model)

    bf = BlockFisher()
    for name, m in mods.items():
        d_in = m.weight.shape[0] if m.__class__.__name__ == "Conv1D" else m.weight.shape[1]
        bf.cov[name] = torch.zeros(d_in, d_in, device=device)
        bf.count[name] = 0

    handles = []

    def make_hook(name: str):
        def hook(module, inp, out):
            a = inp[0] if isinstance(inp, tuple) else inp
            a = a.reshape(-1, a.shape[-1]).to(device)          # [tokens, d_in]
            bf.cov[name] += a.transpose(0, 1) @ a               # [d_in, d_in]
            bf.count[name] += a.shape[0]
        return hook

    for name, m in mods.items():
        handles.append(m.register_forward_hook(make_hook(name)))

    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_samples, desc="BlockFisher", unit="sample")
    except ImportError:
        pbar = None

    seen = 0
    with torch.no_grad():
        for batch in dataloader:
            if seen >= n_samples:
                break
            input_ids = batch["input_ids"].to(device)
            model(input_ids=input_ids)
            bs = input_ids.shape[0]
            seen += bs
            if pbar is not None:
                pbar.update(bs)

    if pbar is not None:
        pbar.close()
    for h in handles:
        h.remove()

    return bf.finalize()


# ---------------------------------------------------------------------------
# Rotated quantization: NQP-Fisher, NQP-random, and identity baseline
# ---------------------------------------------------------------------------

def _weight_matrix(module: nn.Module) -> tuple[Tensor, bool]:
    """
    Return (W as [d_out, d_in], is_conv1d). HF Conv1D stores weight as [d_in, d_out],
    so we transpose to a common [d_out, d_in] convention for rotation math.
    """
    is_conv1d = module.__class__.__name__ == "Conv1D"
    w = module.weight.data
    return (w.t().contiguous(), True) if is_conv1d else (w, False)


def _store_weight(module: nn.Module, w_out_in: Tensor, is_conv1d: bool) -> None:
    """Write back a [d_out, d_in] matrix into the module, restoring layout."""
    module.weight.data = w_out_in.t().contiguous() if is_conv1d else w_out_in


def quantize_rotated(
    W: Tensor,
    U: Tensor,
    bits: int = 4,
) -> Tensor:
    """
    Quantize W in the basis given by orthogonal U, then rotate back.

        W̃ = W U            (rotate columns into the U-basis)
        Q̂ = per-column symmetric quant of W̃   (scale = max|col| / n_levels)
        Ŵ = Q̂ Uᵀ           (rotate back)

    U = Fisher eigenvectors → NQP.  U = random orthogonal → QuIP-style.  U = I → RTN.
    """
    W_rot = W @ U                                    # [d_out, d_in]
    n_levels = 2 ** (bits - 1) - 1
    # Per-column scale (each rotated input dimension gets its own grid)
    scale = W_rot.abs().amax(dim=0, keepdim=True) / n_levels    # [1, d_in]
    scale = scale.clamp(min=1e-9).expand_as(W_rot)
    W_q = quantize_symmetric(W_rot, scale, bits)
    return W_q @ U.t()                               # rotate back → [d_out, d_in]


def fisher_eig(F: Tensor) -> tuple[Tensor, Tensor]:
    """Return (eigvals, U) with F = U Λ Uᵀ (symmetric eig). Columns of U = eigenvectors."""
    F = 0.5 * (F + F.t())                            # symmetrize for numerical safety
    # Add tiny ridge for positive-definiteness (F is PSD but may be rank-deficient)
    ridge = 1e-6 * torch.diag(F).mean().clamp(min=1e-12)
    eigvals, eigvecs = torch.linalg.eigh(F + ridge * torch.eye(F.shape[0], device=F.device))
    return eigvals.clamp(min=1e-12), eigvecs


def fisher_eigenbasis(F: Tensor) -> Tensor:
    """Backward-compat: return only U."""
    return fisher_eig(F)[1]


def quantize_rotated_fisher_scale(
    W: Tensor,
    eigvals: Tensor,
    U: Tensor,
    bits: int = 4,
) -> Tensor:
    """
    NQP done right (formalism §4): rotate into Fisher eigenbasis AND set the per-column
    grid from the eigenvalues, not from max|col|.

    Component i in the natural basis has expected variance ∝ λ_i^{-1}; the standard
    deviation is ∝ λ_i^{-1/2}. We allocate the grid so each column's scale tracks both
    its own range AND its Fisher sensitivity:

        s_i = (max|W̃_{:,i}|) / n_levels  *  r_i,    r_i = (λ_bar / λ_i)^{1/2} (normalized)

    High λ_i (sensitive direction) → r_i < 1 → finer grid. The r_i is normalized to
    mean 1 and clamped, so total range is preserved (same anchor as the naive version)
    but bits are *redistributed* toward sensitive directions — the §4 active ingredient.
    """
    W_rot = W @ U                                            # [d_out, d_in]
    n_levels = 2 ** (bits - 1) - 1
    base = W_rot.abs().amax(dim=0, keepdim=True) / n_levels  # [1, d_in], per-column anchor

    lam_norm = eigvals / eigvals.mean().clamp(min=1e-12)
    r = lam_norm.rsqrt()                                     # λ_i^{-1/2}, up to constant
    r = r / r.mean().clamp(min=1e-12)                        # normalize mean→1
    r = r.clamp(0.25, 4.0).unsqueeze(0)                      # [1, d_in]

    scale = (base * r).clamp(min=1e-9).expand_as(W_rot)
    W_q = quantize_symmetric(W_rot, scale, bits)
    return W_q @ U.t()


def random_orthogonal(d: int, device, seed: int = 0) -> Tensor:
    """Random orthogonal matrix via QR of a Gaussian (Haar-ish). QuIP baseline."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(d, d, generator=g).to(device)
    Q, R = torch.linalg.qr(A)
    # Fix signs so Q is a proper rotation (det handling not needed for our use)
    Q *= torch.sign(torch.diag(R)).unsqueeze(0)
    return Q


# ---------------------------------------------------------------------------
# A-G4 comparison
# ---------------------------------------------------------------------------

@dataclass
class RotationReport:
    name: str
    err_identity: float    # U = I (RTN)
    err_random: float      # U = random orthogonal (QuIP)
    err_fisher: float      # U = Fisher eigenbasis, naive max|col| scale
    err_fisher_lam: float  # U = Fisher eigenbasis, λ-derived scale (NQP §4, the real one)

    @property
    def nqp_beats_random(self) -> bool:
        return self.err_fisher_lam < self.err_random

    @property
    def nqp_beats_rtn(self) -> bool:
        return self.err_fisher_lam < self.err_identity

    def __repr__(self) -> str:
        best = min(self.err_identity, self.err_random, self.err_fisher, self.err_fisher_lam)
        winner = {
            self.err_identity: "RTN",
            self.err_random: "rand",
            self.err_fisher: "fish",
            self.err_fisher_lam: "NQP*",
        }[best]
        return (
            f"{self.name:38s}  rtn={self.err_identity:.4e}  rand={self.err_random:.4e}  "
            f"fish={self.err_fisher:.4e}  NQP*={self.err_fisher_lam:.4e}  [{winner}]"
        )


def run_ag4(
    n_calib: int = 128,
    bits: int = 4,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    GATE A-G4 — Does the Fisher eigenbasis beat a random orthogonal basis?

    For each linear layer, quantize at `bits` in three bases (I / random / Fisher)
    and compare reconstruction error ||W - Ŵ||²/||W||². The decisive metric is
    how often Fisher beats random: if it's not clearly > 50%, NQP ≈ QuIP and the
    Fisher structure adds nothing.

    Returns dict with per-layer reports and aggregate verdict.
    """
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        from datasets import load_dataset
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise ImportError("requires transformers, datasets, torch") from e

    torch.manual_seed(seed)
    if str(device) == "cpu":
        import os
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    print("[A-G4] Loading GPT-2 small...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    print("[A-G4] Loading WikiText-103 calibration...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    subset = ds["train"].select(range(min(n_calib * 8, len(ds["train"]))))

    def tok_fn(ex):
        return tok(ex["text"], truncation=True, max_length=512, padding="max_length")

    tokenized = subset.map(tok_fn, batched=True, remove_columns=["text"])
    tokenized.set_format("torch")
    loader = DataLoader(tokenized, batch_size=4, shuffle=False)

    print(f"[A-G4] Estimating block Fisher ({n_calib} samples)...")
    fishers = estimate_block_fisher(model, loader, n_samples=n_calib, device=device)

    print(f"[A-G4] Comparing rotations at {bits} bits...\n")
    mods = _linear_modules(model)
    reports = []
    with torch.no_grad():
        for name, m in mods.items():
            if name not in fishers:
                continue
            W, is_conv = _weight_matrix(m)            # [d_out, d_in]
            d_in = W.shape[1]
            F = fishers[name]

            eigvals, U_fisher = fisher_eig(F)
            U_rand = random_orthogonal(d_in, device, seed=seed)
            I = torch.eye(d_in, device=W.device)

            err_id = quant_error_l2(W, quantize_rotated(W, I, bits))
            err_rand = quant_error_l2(W, quantize_rotated(W, U_rand, bits))
            err_fish = quant_error_l2(W, quantize_rotated(W, U_fisher, bits))
            err_fish_lam = quant_error_l2(W, quantize_rotated_fisher_scale(W, eigvals, U_fisher, bits))

            r = RotationReport(name, err_id, err_rand, err_fish, err_fish_lam)
            reports.append(r)
            print(" ", r)

    # ── Aggregate verdict (NQP* = Fisher rotation + λ-derived scale, the §4 method) ──
    n = len(reports)
    n_nqp_beats_rand = sum(r.nqp_beats_random for r in reports)
    n_nqp_beats_rtn = sum(r.nqp_beats_rtn for r in reports)
    mean_id = sum(r.err_identity for r in reports) / max(n, 1)
    mean_rand = sum(r.err_random for r in reports) / max(n, 1)
    mean_fish = sum(r.err_fisher for r in reports) / max(n, 1)
    mean_nqp = sum(r.err_fisher_lam for r in reports) / max(n, 1)

    print(f"\n{'='*70}\n[A-G4 v2 VERDICT]  bits={bits}, n_calib={n_calib}\n{'='*70}")
    print(f"  NQP* (Fisher+λ-scale) beats RANDOM:  {n_nqp_beats_rand}/{n} layers")
    print(f"  NQP* beats RTN (identity):           {n_nqp_beats_rtn}/{n} layers")
    print(f"  mean err — rtn={mean_id:.4e}  random={mean_rand:.4e}  "
          f"fish-naive={mean_fish:.4e}  NQP*={mean_nqp:.4e}")
    # Decisive comparison: does the real NQP method beat BOTH random and RTN?
    frac_rand = n_nqp_beats_rand / max(n, 1)
    beats_rtn_mean = mean_nqp < mean_id
    beats_rand_mean = mean_nqp < mean_rand
    if frac_rand >= 0.5 and beats_rtn_mean:
        print(f"  => PASS: NQP* beats random in {frac_rand:.0%} AND beats RTN in mean — "
              f"Fisher structure has content. Proceed to GPTQ comparison (A-G3).")
    elif beats_rand_mean and not beats_rtn_mean:
        print(f"  => PARTIAL: NQP* beats random in mean but not RTN — rotation still costs "
              f"more than it saves at {bits}-bit. Try lower bits or per-eigval bit allocation.")
    else:
        print(f"  => FAIL: NQP* beats random only {frac_rand:.0%} — Fisher structure does not "
              f"help with this quantizer. Reconsider the approach (see ROADMAP).")

    return {
        "reports": reports,
        "n_nqp_beats_random": n_nqp_beats_rand,
        "n_nqp_beats_rtn": n_nqp_beats_rtn,
        "n_layers": n,
        "mean_err": {"identity": mean_id, "random": mean_rand,
                     "fisher_naive": mean_fish, "nqp": mean_nqp},
        "pass": frac_rand >= 0.5 and beats_rtn_mean,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="NQP Camino A — block Fisher rotation (A-G4 gate)")
    p.add_argument("--bits", type=int, default=4, help="Quantization bits (4 = signal regime)")
    p.add_argument("--n-calib", type=int, default=128, help="Calibration samples")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    run_ag4(n_calib=args.n_calib, bits=args.bits, device=args.device)
