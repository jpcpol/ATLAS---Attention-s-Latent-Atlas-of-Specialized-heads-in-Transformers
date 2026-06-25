"""
NQP — Uncertainty principle experiment (EXP-U01).

Tests conjecture NQP-U1a: do the eigenbases of the weight-Fisher F_W and the
activation-Fisher G_A *commute*? If [P_W, P_A] != 0, weight-quantization and
activation-quantization cannot be jointly optimized — a genuine uncertainty
relation (theory/uncertainty_principle.md).

This is the part of NQP with real quantum structure: non-commuting observables,
not the metaphorical "measure in the Hamiltonian basis" of the quantization side
(which collapsed to GPTQ+AWQ+QuIP — see ROADMAP.md).

Both Fishers live in R^{d_in} for a linear layer y = W a:
  G_A = E[a aᵀ]                         (input second moment; what GPTQ uses)
  F_W = E[ (∂L/∂W)ᵀ (∂L/∂W) ] projected to the input side, approximated here by
        the column-covariance of the weight-gradient, which shares the d_in space.

We compare their eigenbases via principal angles between dominant subspaces and a
normalized commutator norm.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Iterable

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch import Tensor

from fisher import _should_quantize
from fisher_block import _linear_modules, _weight_matrix, fisher_eig


# ---------------------------------------------------------------------------
# Estimate both Fishers in the shared d_in space
# ---------------------------------------------------------------------------

def estimate_both_fishers(
    model: nn.Module,
    dataloader: Iterable,
    n_samples: int = 64,
    device: str | torch.device = "cpu",
) -> dict[str, tuple[Tensor, Tensor]]:
    """
    For each linear layer, return (F_W, G_A), both d_in × d_in.

    G_A = E[a aᵀ]                  via forward hook on layer input.
    F_W ≈ E[ gᵀ g ]  where g = ∂L/∂(layer output) ⊗ a is the weight gradient;
         projecting to the input side gives the d_in×d_in column-covariance of
         the weight gradient: (∂L/∂W)ᵀ(∂L/∂W) ∈ R^{d_in×d_in}.

    Both share the d_in input space, so their eigenbases are directly comparable.
    """
    model.eval()
    model.to(device)
    mods = _linear_modules(model)

    G_acc: dict[str, Tensor] = {}
    n_act: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(module, inp, out):
            a = (inp[0] if isinstance(inp, tuple) else inp).reshape(-1, inp[0].shape[-1]).to(device)
            if name not in G_acc:
                G_acc[name] = torch.zeros(a.shape[1], a.shape[1], device=device)
                n_act[name] = 0
            G_acc[name] += a.t() @ a
            n_act[name] += a.shape[0]
        return hook

    for name, m in mods.items():
        handles.append(m.register_forward_hook(make_hook(name)))

    FW_acc: dict[str, Tensor] = {}
    n_grad = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_samples, desc="BothFishers", unit="sample")
    except ImportError:
        pbar = None

    seen = 0
    for batch in dataloader:
        if seen >= n_samples:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", input_ids).to(device)

        model.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()

        for name, m in mods.items():
            if m.weight.grad is None:
                continue
            gW, _ = _weight_matrix_grad(m)        # [d_out, d_in]
            col_cov = gW.t() @ gW                 # [d_in, d_in]
            if name not in FW_acc:
                FW_acc[name] = torch.zeros_like(col_cov)
            FW_acc[name] += col_cov

        n_grad += 1
        bs = input_ids.shape[0]
        seen += bs
        if pbar is not None:
            pbar.update(bs)

    if pbar is not None:
        pbar.close()
    for h in handles:
        h.remove()

    result = {}
    for name in mods:
        if name in G_acc and name in FW_acc:
            G = G_acc[name] / max(n_act[name], 1)
            F = FW_acc[name] / max(n_grad, 1)
            result[name] = (F, G)
    return result


def _weight_matrix_grad(module: nn.Module) -> tuple[Tensor, bool]:
    """Like _weight_matrix but for .grad, returning [d_out, d_in]."""
    is_conv1d = module.__class__.__name__ == "Conv1D"
    g = module.weight.grad
    return (g.t().contiguous(), True) if is_conv1d else (g, False)


# ---------------------------------------------------------------------------
# Commutator / subspace-angle metrics
# ---------------------------------------------------------------------------

def principal_angles(U: Tensor, V: Tensor, k: int) -> Tensor:
    """
    Principal angles (radians) between the top-k subspaces spanned by columns of U, V.
    0 = aligned subspaces (commute on that subspace), π/2 = orthogonal (max conflict).
    """
    Uk = U[:, :k]
    Vk = V[:, :k]
    # SVD of UkᵀVk → singular values = cosines of principal angles
    s = torch.linalg.svdvals(Uk.t() @ Vk).clamp(-1.0, 1.0)
    return torch.arccos(s)


def commutator_norm(P_W: Tensor, P_A: Tensor) -> float:
    """
    Normalized Frobenius norm of [P_W, P_A] = P_W P_A - P_A P_W over the FULL bases.

    WARNING (EXP-U01, 2026-06-24): in high dimension (d=768) this saturates to ~1.0
    for ANY pair of non-identical orthogonal bases — a random-orthogonal pair also
    gives 1.0000. So the full-rank commutator does NOT discriminate structured-but-
    non-commuting from random. Use `principal_angles` on the dominant subspace instead;
    that one separates GPT-2 (48.8°) from random (83°) from commuting (0°).

    Kept for completeness, not as the decisive metric.
    """
    C = P_W @ P_A - P_A @ P_W
    d = P_W.shape[0]
    return (C.norm().item()) / math.sqrt(2 * d)


def random_baseline_angle(d: int, k: int, n_seeds: int = 5) -> float:
    """
    Mean principal angle (deg) between two random orthogonal top-k subspaces in R^d.
    This is the "no shared structure" reference: observed angles well BELOW this mean
    structure beyond chance; angles AT this value mean the bases are effectively random.
    """
    import statistics
    vals = []
    for s in range(n_seeds):
        g = torch.Generator().manual_seed(1000 + s)
        Q1, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        Q2, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
        vals.append(math.degrees(principal_angles(Q1, Q2, k).mean().item()))
    return statistics.mean(vals)


@dataclass
class UncertaintyReport:
    name: str
    d: int
    commutator: float          # normalized ||[P_W, P_A]||_F
    mean_angle_top: float      # mean principal angle (deg) over dominant subspace
    spec_overlap: float        # |<top eigvec_W, top eigvec_A>| (1 = aligned)

    def __repr__(self) -> str:
        return (
            f"{self.name:38s}  d={self.d:4d}  "
            f"||[Pw,Pa]||={self.commutator:.4f}  "
            f"angle_top={self.mean_angle_top:5.1f}°  "
            f"top_overlap={self.spec_overlap:.3f}"
        )


def run_exp_u01(
    n_calib: int = 64,
    top_k: int = 16,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    EXP-U01 — Measure [P_W, P_A] per layer in GPT-2. Tests NQP-U1a.

    If commutator norms are significantly > 0 across layers, the weight- and
    activation-Fisher bases do NOT commute → an uncertainty relation exists.
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

    print("[EXP-U01] Loading GPT-2 small...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    print("[EXP-U01] Loading WikiText-103 calibration...")
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

    print(f"[EXP-U01] Estimating F_W and G_A ({n_calib} samples)...")
    fishers = estimate_both_fishers(model, loader, n_samples=n_calib, device=device)

    print(f"[EXP-U01] Measuring commutators (top_k={top_k})...\n")
    reports = []
    for name, (F_W, G_A) in fishers.items():
        ev_w, P_W = fisher_eig(F_W)
        ev_a, P_A = fisher_eig(G_A)
        # eigh returns ascending; flip so column 0 = dominant direction
        P_W = P_W.flip(1)
        P_A = P_A.flip(1)

        comm = commutator_norm(P_W, P_A)
        angles = principal_angles(P_W, P_A, k=min(top_k, P_W.shape[0]))
        mean_angle = math.degrees(angles.mean().item())
        overlap = abs((P_W[:, 0] @ P_A[:, 0]).item())

        r = UncertaintyReport(name, P_W.shape[0], comm, mean_angle, overlap)
        reports.append(r)
        print(" ", r)

    # ── Verdict ────────────────────────────────────────────────────────────
    n = len(reports)
    mean_comm = sum(r.commutator for r in reports) / max(n, 1)
    mean_angle = sum(r.mean_angle_top for r in reports) / max(n, 1)
    mean_overlap = sum(r.spec_overlap for r in reports) / max(n, 1)

    # The decisive reference: angle between two RANDOM subspaces of the same size.
    d_ref = reports[0].d if reports else 768
    rand_angle = random_baseline_angle(d_ref, min(top_k, d_ref))

    print(f"\n{'='*70}\n[EXP-U01 VERDICT]  test NQP-U1a (do P_W, P_A commute?)\n{'='*70}")
    print(f"  mean principal angle (top-{top_k}):  {mean_angle:.1f}°")
    print(f"  RANDOM baseline angle:           {rand_angle:.1f}°   (no shared structure)")
    print(f"  COMMUTING reference:              0.0°   (shared eigenbasis)")
    print(f"  mean top-eigvec overlap:         {mean_overlap:.3f}   "
          f"(random ≈ {1.0/math.sqrt(d_ref):.3f})")
    print(f"  [full-rank ||[P,P]||={mean_comm:.3f} — saturates in high-d, not decisive]")
    # Structured-but-non-commuting = significantly below random AND significantly above 0.
    below_random = mean_angle < 0.75 * rand_angle      # clearly more aligned than chance
    above_commute = mean_angle > 15.0                  # clearly not commuting
    if below_random and above_commute:
        print(f"  => NQP-U1a SUPPORTED: {mean_angle:.0f}° sits between commuting (0°) and "
              f"random ({rand_angle:.0f}°). Bases share structure yet do NOT commute — the "
              f"precondition for a weight/activation uncertainty relation holds. "
              f"Proceed to EXP-U02 (Pareto frontier).")
    elif not above_commute:
        print(f"  => NQP-U1a FALSIFIED: {mean_angle:.0f}° ≈ 0° — bases commute, no uncertainty.")
    else:
        print(f"  => NQP-U1a WEAK: {mean_angle:.0f}° ≈ random ({rand_angle:.0f}°) — bases are "
              f"effectively unrelated, the conflict is trivial (any two bases would show it).")

    return {
        "reports": reports,
        "mean_commutator": mean_comm,
        "mean_angle_deg": mean_angle,
        "mean_overlap": mean_overlap,
        "n_layers": n,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="NQP EXP-U01 — uncertainty principle test")
    p.add_argument("--n-calib", type=int, default=64)
    p.add_argument("--top-k", type=int, default=16, help="Dominant subspace dim for angles")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    run_exp_u01(n_calib=args.n_calib, top_k=args.top_k, device=args.device)
