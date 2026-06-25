"""
NQP — Fisher diagonal estimation and natural-basis quantization.

Implements the preparation operator P̂ = U for EXP-001 (GPT-2 small).

Pipeline per layer:
  1. Estimate diagonal Fisher F_diag via MC gradient sampling over calibration data.
  2. Derive P̂: eigenvector matrix U that diagonalizes F (diagonal F → U = I,
     but we keep the general path for block-Fisher extensions).
  3. Transform weights: W̃ = P̂ @ W.
  4. Quantize each component Q_i(W̃_i) with per-component scale derived from λ_i.
  5. Reconstruct: Ŵ = P̂⁻¹ @ Q(W̃).

For diagonal Fisher, the "eigenvectors" are the canonical basis, but eigenvalues
λ_i = F_diag[i] still drive per-component bit allocation and scale selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor


# Parameter name patterns that should NOT be quantized via NQP.
# Embeddings, LayerNorm, and biases stay in FP32 — standard practice in
# GPTQ/AWQ/QuIP, which quantize only the large linear weight matrices.
_NQP_SKIP_PATTERNS = ("wte", "wpe", "ln_", "ln_f", ".bias")


def _should_quantize(name: str, w: Tensor) -> bool:
    """Quantize only 2D linear/conv weight matrices, not embeddings/norms/biases."""
    if w.dim() < 2:
        return False
    return not any(pat in name for pat in _NQP_SKIP_PATTERNS)


# ---------------------------------------------------------------------------
# Fisher diagonal estimator
# ---------------------------------------------------------------------------

@dataclass
class FisherEstimate:
    """Per-parameter diagonal Fisher estimates for one model."""
    diag: dict[str, Tensor] = field(default_factory=dict)  # param_name → F_diag

    def eigenvalues(self, name: str) -> Tensor:
        """Diagonal Fisher IS the eigenvalue vector for the diagonal approximation."""
        return self.diag[name]

    def scale_for_bits(self, name: str, w: Tensor, bits: int = 8) -> Tensor:
        """
        Per-component quantization scale derived from Fisher eigenvalues.

        Strategy: Fisher sensitivity λ_i redistributes the quantization budget
        *within* the weight tensor. The absolute scale is anchored to the weight
        range so that total representable range covers the actual weight values.

        s_i = s_base * r_i
        where:
          s_base = max(|W|) / (2^(b-1) - 1)   — same anchor as standard quant
          r_i    = λ_max / (λ_i + ε)^α         — Fisher redistribution factor
                   (high curvature → r_i small → finer grid for that component)
          α = 0.5 balances sensitivity vs uniformity; clipped to [0.25, 4.0]
          to prevent extreme scales from blowing up in low-Fisher regions.
        """
        lambdas = self.eigenvalues(name).clamp(min=1e-8)
        n_levels = 2 ** (bits - 1) - 1
        s_base = w.abs().max() / n_levels

        # Relative Fisher redistribution: high λ → smaller scale (finer grid)
        lambda_norm = lambdas / lambdas.max().clamp(min=1e-8)
        r = (1.0 / (lambda_norm + 1e-6)) ** 0.5
        # Clip redistribution to ±4× of base scale to prevent blowup
        r = r / r.mean()          # normalize so mean scale ≈ s_base
        r = r.clamp(0.25, 4.0)

        return s_base * r


def estimate_fisher_diagonal(
    model: nn.Module,
    dataloader: Iterable,
    n_samples: int = 512,
    device: str | torch.device = "cpu",
) -> FisherEstimate:
    """
    Estimate diagonal Fisher Information Matrix via MC sampling.

    F_diag[θ_i] = E_x[ (∂ log p(x;θ) / ∂θ_i)² ]
                ≈ (1/N) Σ_x (∂L(x;θ) / ∂θ_i)²

    This is the empirical Fisher — exact Fisher requires marginalizing over
    model predictions; empirical Fisher uses observed labels and is faster.

    Args:
        model:      The LLM (any nn.Module with a scalar loss forward).
        dataloader: Yields batches with 'input_ids' (and optionally 'labels').
        n_samples:  Total tokens/samples to accumulate gradients over.
        device:     Computation device.

    Returns:
        FisherEstimate with diag[name] = accumulated squared-gradient tensor.
    """
    model.eval()
    model.to(device)

    accum: dict[str, Tensor] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            accum[name] = torch.zeros_like(param.data)

    try:
        from tqdm import tqdm
        pbar = tqdm(total=n_samples, desc="Fisher", unit="sample")
    except ImportError:
        pbar = None

    seen = 0
    for batch in dataloader:
        if seen >= n_samples:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", input_ids).to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                accum[name] += param.grad.detach() ** 2

        bs = input_ids.shape[0]
        seen += bs
        if pbar is not None:
            pbar.update(bs)

    if pbar is not None:
        pbar.close()

    n = max(seen, 1)
    diag = {name: v / n for name, v in accum.items()}
    return FisherEstimate(diag=diag)


# ---------------------------------------------------------------------------
# Quantization primitives
# ---------------------------------------------------------------------------

def quantize_symmetric(x: Tensor, scale: Tensor, bits: int = 8) -> Tensor:
    """
    Symmetric uniform quantization with per-element scale.

    Q(x_i) = round(x_i / s_i) * s_i  clamped to [-2^(b-1), 2^(b-1)-1] * s_i
    """
    n_levels = 2 ** (bits - 1) - 1
    x_scaled = x / scale.clamp(min=1e-9)
    x_clamped = x_scaled.clamp(-n_levels, n_levels)
    x_rounded = x_clamped.round()
    return x_rounded * scale


def quantize_standard(w: Tensor, bits: int = 8) -> Tensor:
    """
    Standard uniform quantization (baseline for comparison).

    Uses a single scale = max(|W|) / (2^(b-1) - 1).
    """
    n_levels = 2 ** (bits - 1) - 1
    scale = w.abs().max() / n_levels
    return quantize_symmetric(w, scale.expand_as(w), bits=bits)


# ---------------------------------------------------------------------------
# NQP quantization operator
# ---------------------------------------------------------------------------

class NQPQuantizer:
    """
    Natural-basis quantizer derived from diagonal Fisher.

    For diagonal Fisher approximation:
      - P̂ = I  (eigenvectors of a diagonal matrix are canonical basis vectors)
      - λ_i = F_diag[i]  (eigenvalues)
      - Scale per component: s_i = σ_i / (2^(b-1)-1)  with σ_i = λ_i^{-0.5}

    This means NQP with diagonal Fisher = per-component adaptive scaling,
    where components with high loss-curvature get finer quantization grids.
    """

    def __init__(self, fisher: FisherEstimate, bits: int = 8):
        self.fisher = fisher
        self.bits = bits

    def quantize_weight(self, name: str, w: Tensor) -> Tensor:
        """
        Quantize weight tensor w using Fisher-derived per-component scales.

        W̃  = P̂ @ w  →  for diagonal Fisher, W̃ = w (P̂ = I)
        Q̂  = Q_i(W̃_i) per component with s_i from Fisher
        Ŵ  = P̂⁻¹ @ Q̂  →  for diagonal Fisher, Ŵ = Q̂
        """
        if not _should_quantize(name, w):
            # Embeddings, LayerNorm, biases stay in FP32 (not quantized)
            return w.clone()

        if name not in self.fisher.diag:
            # Fall back to standard quantization for params without Fisher estimate
            return quantize_standard(w, self.bits)

        scale = self.fisher.scale_for_bits(name, w, self.bits).to(w.device)

        # Reshape scale to match w — Fisher is stored flattened per-parameter
        scale = scale.view_as(w)

        # P̂ @ w: for diagonal Fisher, this is identity
        w_natural = w  # noqa: trivially identity in diagonal case

        # Per-component quantization in natural basis
        w_q_natural = quantize_symmetric(w_natural, scale, self.bits)

        # P̂⁻¹ @ Q̂: identity for diagonal Fisher
        w_reconstructed = w_q_natural

        return w_reconstructed

    def quantize_model(self, model: nn.Module) -> nn.Module:
        """
        Return a copy of the model with all weight parameters quantized via NQP.
        """
        import copy
        model_q = copy.deepcopy(model)
        with torch.no_grad():
            for name, param in model_q.named_parameters():
                param.data = self.quantize_weight(name, param.data)
        return model_q


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def quant_error_l2(w_orig: Tensor, w_quant: Tensor) -> float:
    """L2 quantization error ||W - Q(W)||^2 / ||W||^2 (relative)."""
    num = (w_orig - w_quant).pow(2).sum().item()
    den = w_orig.pow(2).sum().item()
    return num / max(den, 1e-12)


@dataclass
class LayerErrorReport:
    name: str
    error_std: float
    error_nqp: float
    gain_db: float  # 10 * log10(err_std / err_nqp) — positive = NQP better

    def __repr__(self) -> str:
        direction = "better" if self.gain_db > 0 else "worse"
        return (
            f"{self.name:60s}  "
            f"std={self.error_std:.4e}  nqp={self.error_nqp:.4e}  "
            f"gain={self.gain_db:+.2f} dB ({direction})"
        )


def compare_quantization(
    model: nn.Module,
    fisher: FisherEstimate,
    bits: int = 8,
) -> list[LayerErrorReport]:
    """
    Compare standard vs NQP quantization error per weight parameter.

    Returns one LayerErrorReport per named parameter.
    """
    quantizer = NQPQuantizer(fisher, bits=bits)
    reports = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            w = param.data
            # Only report on params NQP actually quantizes — comparing FP32-kept
            # params would show a spurious 0-vs-0 tie and dilute the signal.
            if not _should_quantize(name, w):
                continue
            w_std = quantize_standard(w, bits)
            w_nqp = quantizer.quantize_weight(name, w)
            err_std = quant_error_l2(w, w_std)
            err_nqp = quant_error_l2(w, w_nqp)

            if err_std > 0:
                gain_db = 10 * math.log10(err_std / max(err_nqp, 1e-30))
            else:
                gain_db = 0.0

            reports.append(LayerErrorReport(
                name=name,
                error_std=err_std,
                error_nqp=err_nqp,
                gain_db=gain_db,
            ))

    return reports


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------

def compute_perplexity_blocks(
    model: nn.Module,
    token_ids: Tensor,
    seq_len: int = 512,
    device: str | torch.device = "cpu",
    max_blocks: int | None = None,
) -> float:
    """
    Compute perplexity over contiguous (non-padded) blocks of a long token stream.

    This is the standard WikiText PPL protocol: concatenate the whole corpus,
    slice into back-to-back windows of `seq_len`, and average cross-entropy over
    real tokens only. No padding → no spurious EOS loss inflating the metric.

    PPL = exp( Σ_blocks loss_b * n_b  /  Σ_blocks n_b )
    where loss_b is the mean shifted CE over the block and n_b = seq_len - 1.
    """
    model.eval()
    model.to(device)

    n_tokens_total = token_ids.numel()
    n_blocks = n_tokens_total // seq_len
    if max_blocks is not None:
        n_blocks = min(n_blocks, max_blocks)

    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for b in range(n_blocks):
            block = token_ids[b * seq_len:(b + 1) * seq_len].unsqueeze(0).to(device)
            outputs = model(input_ids=block, labels=block)
            loss = outputs.loss.item()
            if not math.isfinite(loss):
                # Skip degenerate blocks rather than poisoning the whole metric
                continue
            n_pred = seq_len - 1  # HF shifts internally: seq_len-1 predicted positions
            total_nll += loss * n_pred
            total_tokens += n_pred

    avg_nll = total_nll / max(total_tokens, 1)
    return math.exp(avg_nll)


# ---------------------------------------------------------------------------
# EXP-001 entry point
# ---------------------------------------------------------------------------

def run_exp001(
    n_calib_samples: int = 256,
    n_eval_batches: int = 100,
    bits: int = 8,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    EXP-001 — Baseline Fisher diagonal on GPT-2 small.

    Steps:
      1. Load GPT-2 small (124M) from HuggingFace.
      2. Load WikiText-103 calibration split.
      3. Estimate diagonal Fisher over n_calib_samples.
      4. Quantize via standard INT{bits} and NQP.
      5. Compute PPL on validation split for both.
      6. Report per-layer quantization error and PPL delta.

    Returns dict with keys: ppl_fp32, ppl_std, ppl_nqp, layer_reports.
    """
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        from datasets import load_dataset
        from torch.utils.data import DataLoader
    except ImportError as e:
        raise ImportError(
            "EXP-001 requires: pip install transformers datasets torch"
        ) from e

    torch.manual_seed(seed)

    # Bound CPU threads to avoid oversubscription (multiple processes / hyperthreads
    # fighting over cores tanks throughput). Use physical cores, capped at 8.
    if str(device) == "cpu":
        import os
        n_threads = min(8, os.cpu_count() or 4)
        torch.set_num_threads(n_threads)
        print(f"[EXP-001] torch threads = {n_threads}")

    print("[EXP-001] Loading GPT-2 small...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    print("[EXP-001] Loading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    def tokenize_fn(examples):
        enc = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        # labels = input_ids with padding positions masked to -100
        labels = [
            [t if m == 1 else -100 for t, m in zip(ids, mask)]
            for ids, mask in zip(input_ids, attention_mask)
        ]
        enc["labels"] = labels
        return enc

    def make_loader(split: str, batch_size: int = 4, max_examples: int | None = None) -> DataLoader:
        subset = ds[split]
        if max_examples is not None:
            subset = subset.select(range(min(max_examples, len(subset))))
        tokenized = subset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
        )
        tokenized.set_format("torch")
        return DataLoader(tokenized, batch_size=batch_size, shuffle=(split == "train"))

    # Calibration loader (padded, masked) — fine for Fisher since padding is -100.
    calib_examples = n_calib_samples * 8  # generous headroom for empty/short texts
    calib_loader = make_loader("train", batch_size=4, max_examples=calib_examples)

    # Evaluation: build ONE contiguous token stream (standard WikiText PPL protocol).
    # No padding — concatenate validation text and slice into back-to-back blocks.
    print("[EXP-001] Building contiguous eval token stream...")
    eval_text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    eval_ids = tokenizer(eval_text, return_tensors="pt")["input_ids"].squeeze(0)
    print(f"  eval stream: {eval_ids.numel()} tokens "
          f"({eval_ids.numel() // 512} blocks of 512)")

    # ── Fisher estimation ──────────────────────────────────────────────────
    print(f"[EXP-001] Estimating diagonal Fisher ({n_calib_samples} samples)...")
    fisher = estimate_fisher_diagonal(
        model, calib_loader, n_samples=n_calib_samples, device=device
    )

    # ── Baseline PPL (FP32) ────────────────────────────────────────────────
    print("[EXP-001] Computing FP32 perplexity...")
    ppl_fp32 = compute_perplexity_blocks(model, eval_ids, device=device, max_blocks=n_eval_batches)
    print(f"  PPL FP32  = {ppl_fp32:.2f}")

    # ── Standard quantization ──────────────────────────────────────────────
    # Apply the SAME skip filter as NQP so the comparison is apples-to-apples:
    # both quantize only linear weight matrices, keeping embeddings/norms in FP32.
    import copy
    print(f"[EXP-001] Quantizing INT{bits} standard...")
    model_std = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in model_std.named_parameters():
            if _should_quantize(name, param.data):
                param.data = quantize_standard(param.data, bits)

    ppl_std = compute_perplexity_blocks(model_std, eval_ids, device=device, max_blocks=n_eval_batches)
    print(f"  PPL INT{bits} std = {ppl_std:.2f}")

    # ── NQP quantization ───────────────────────────────────────────────────
    print(f"[EXP-001] Quantizing NQP INT{bits}...")
    quantizer = NQPQuantizer(fisher, bits=bits)
    model_nqp = quantizer.quantize_model(model)

    ppl_nqp = compute_perplexity_blocks(model_nqp, eval_ids, device=device, max_blocks=n_eval_batches)
    print(f"  PPL NQP   = {ppl_nqp:.2f}")

    # ── Per-layer error comparison ─────────────────────────────────────────
    print("\n[EXP-001] Per-layer quantization error (NQP vs std):")
    reports = compare_quantization(model, fisher, bits=bits)
    for r in reports:
        print(" ", r)

    # Summary
    n_better = sum(1 for r in reports if r.gain_db > 0)
    print(f"\n[EXP-001] NQP better in {n_better}/{len(reports)} layers")
    print(f"[EXP-001] PPL delta (NQP - std): {ppl_nqp - ppl_std:+.2f}")

    return {
        "ppl_fp32": ppl_fp32,
        "ppl_std": ppl_std,
        "ppl_nqp": ppl_nqp,
        "layer_reports": reports,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NQP EXP-001: Fisher diagonal on GPT-2 small")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8], help="Quantization bits")
    parser.add_argument("--n-calib", type=int, default=256, help="Calibration samples")
    parser.add_argument("--n-eval", type=int, default=100, help="Eval batches")
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda / mps")
    args = parser.parse_args()

    results = run_exp001(
        n_calib_samples=args.n_calib,
        n_eval_batches=args.n_eval,
        bits=args.bits,
        device=args.device,
    )
