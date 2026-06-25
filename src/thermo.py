"""
NQP-Q — Thermodynamic characterization of transformer attention (EXP-Q01).

Every quantity below derives from the SAME partition function Z that attention
already computes (theory/quantum_transformer_map.md). For one query position with
attention logits s_i = q·k_i/√d over keys i:

  β = 1                       (inverse temperature; the √d is folded into s_i)
  Z = Σ_i e^{β s_i}           partition function (softmax denominator)
  A_i = e^{β s_i} / Z         attention weights (Boltzmann distribution)
  E_i = -s_i                  energy of state i
  ⟨E⟩ = Σ_i A_i E_i           mean energy
  F = -(1/β) log Z            Helmholtz free energy            [GAP-F]
  S_shannon = -Σ A_i log A_i  Shannon/Gibbs entropy of weights
  C = β²(⟨E²⟩ - ⟨E⟩²)         heat capacity = energy variance   [GAP-K]
  T_eff = exp(S_shannon)      effective temperature ~ participation (# keys attended)

Density matrix over the value vectors (mixed state of the attention output):
  ρ = Σ_i A_i |v̂_i⟩⟨v̂_i|     (v̂ = unit-normalized value vectors)   [GAP-A/H]
  purity  = Tr(ρ²)            ∈ [1/d, 1]; 1 = pure (one value), low = spread
  S_vn    = -Tr(ρ log ρ)      von Neumann entropy                 [GAP-H]

All measured per (layer, head) and correlated with prediction entropy / PPL.
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


# ---------------------------------------------------------------------------
# Per-distribution thermodynamic quantities (from attention logits)
# ---------------------------------------------------------------------------

def thermo_from_logits(s: Tensor) -> dict:
    """
    Thermodynamic scalars for one attention distribution.

    s: [..., n_keys] attention logits (already scaled by 1/√d). β folded in (=1).
    Masked-out keys (causal future) carry s = -inf; we handle them explicitly so
    that 0·∞ never poisons the energy moments. Returns tensors shaped [...].
    """
    valid = torch.isfinite(s)                          # [...,n] True for real keys

    # Stable log-sum-exp over valid keys only
    s_safe = torch.where(valid, s, torch.full_like(s, float("-inf")))
    s_max = s_safe.amax(dim=-1, keepdim=True)
    z = torch.where(valid, (s_safe - s_max).exp(), torch.zeros_like(s))
    Zsum = z.sum(dim=-1)                               # [...]
    logZ = s_max.squeeze(-1) + Zsum.log()              # log partition fn
    A = z / Zsum.unsqueeze(-1).clamp(min=1e-30)        # Boltzmann weights [...,n]

    # Energy only on valid keys; masked contribute exactly 0 (not 0·∞=NaN).
    E = torch.where(valid, -s, torch.zeros_like(s))    # energy per state
    meanE = (A * E).sum(dim=-1)                        # ⟨E⟩
    meanE2 = (A * E * E).sum(dim=-1)                   # ⟨E²⟩
    C = (meanE2 - meanE * meanE).clamp(min=0.0)        # heat capacity (β=1)

    F = -logZ                                          # Helmholtz free energy (β=1)
    S = -(A * A.clamp(min=1e-12).log()).sum(dim=-1)    # Shannon entropy
    T_eff = S.exp()                                    # effective temperature (perplexity of A)

    return {"free_energy": F, "mean_E": meanE, "heat_capacity": C,
            "shannon_entropy": S, "T_eff": T_eff}


def density_matrix_metrics(A: Tensor, V: Tensor) -> tuple[Tensor, Tensor]:
    """
    Purity and von Neumann entropy of ρ = Σ_i A_i |v̂_i⟩⟨v̂_i|.

    A: [..., n_keys] attention weights (sum to 1 over last dim).
    V: [..., n_keys, d_head] value vectors.
    Returns (purity, S_vn), each shaped [...].

    ρ is [..., d_head, d_head]; we unit-normalize values so the mixed state lives
    on the unit sphere (pure-state extremes have purity 1).
    """
    Vn = V / V.norm(dim=-1, keepdim=True).clamp(min=1e-9)     # [...,n,d]
    # ρ = Σ_i A_i v̂_i v̂_iᵀ  via weighted outer products
    AV = A.unsqueeze(-1) * Vn                                  # [...,n,d]
    rho = torch.einsum("...nd,...ne->...de", AV, Vn)          # [...,d,d]
    # symmetrize for numerical safety
    rho = 0.5 * (rho + rho.transpose(-1, -2))

    purity = torch.einsum("...de,...ed->...", rho, rho)       # Tr(ρ²)

    # von Neumann entropy via eigenvalues of ρ (PSD, trace 1)
    eig = torch.linalg.eigvalsh(rho).clamp(min=1e-12)         # [...,d]
    eig = eig / eig.sum(dim=-1, keepdim=True)
    S_vn = -(eig * eig.log()).sum(dim=-1)
    return purity, S_vn


# ---------------------------------------------------------------------------
# Hook-based extraction of attention internals from GPT-2
# ---------------------------------------------------------------------------

@dataclass
class HeadStats:
    """Accumulated per-head thermodynamic stats over many query positions."""
    layer: int
    head: int
    free_energy: float = 0.0
    mean_E: float = 0.0
    heat_capacity: float = 0.0
    shannon_entropy: float = 0.0
    T_eff: float = 0.0
    purity: float = 0.0
    S_vn: float = 0.0
    n: int = 0

    def add(self, d: dict, count: int):
        for k in ("free_energy", "mean_E", "heat_capacity",
                  "shannon_entropy", "T_eff", "purity", "S_vn"):
            setattr(self, k, getattr(self, k) + d[k] * count)
        self.n += count

    def finalize(self) -> "HeadStats":
        if self.n:
            for k in ("free_energy", "mean_E", "heat_capacity",
                      "shannon_entropy", "T_eff", "purity", "S_vn"):
                setattr(self, k, getattr(self, k) / self.n)
        return self

    def __repr__(self) -> str:
        return (f"L{self.layer:2d}H{self.head:2d}  "
                f"F={self.free_energy:7.3f}  ⟨E⟩={self.mean_E:7.3f}  "
                f"C={self.heat_capacity:6.3f}  S={self.shannon_entropy:5.3f}  "
                f"Teff={self.T_eff:6.2f}  purity={self.purity:.3f}  Svn={self.S_vn:5.3f}")


def _attn_logits_and_values(model, hidden, layer_idx, attn_module):
    """
    Recompute per-head attention logits s = QKᵀ/√d and value vectors for GPT-2's
    Conv1D attention, given the layer input hidden states [B, T, D].
    Returns (logits [B, H, T, T] causal-masked, values [B, H, T, d_head]).
    """
    B, T, D = hidden.shape
    n_head = attn_module.num_heads if hasattr(attn_module, "num_heads") else attn_module.n_head
    qkv = attn_module.c_attn(hidden)                          # [B,T,3D]
    q, k, v = qkv.split(D, dim=2)
    dh = D // n_head

    def shape(x):
        return x.view(B, T, n_head, dh).permute(0, 2, 1, 3)  # [B,H,T,dh]

    q, k, v = shape(q), shape(k), shape(v)
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(dh)       # [B,H,T,T]
    # causal mask
    mask = torch.tril(torch.ones(T, T, device=hidden.device, dtype=torch.bool))
    logits = logits.masked_fill(~mask, float("-inf"))
    return logits, v


def run_exp_q01(
    n_batches: int = 8,
    device: str = "cpu",
    seed: int = 42,
    min_context: int = 16,
) -> dict:
    """
    EXP-Q01 — measure per-head thermodynamics across GPT-2 on WikiText-103.

    For each layer/head and each query position (with ≥ min_context keys, to avoid
    degenerate short-context distributions), compute the thermodynamic scalars and
    density-matrix metrics, averaged over positions and batches. Also correlate the
    per-position heat capacity / entropy with the model's next-token prediction
    entropy at that position.
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

    print("[EXP-Q01] Loading GPT-2 small...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    print("[EXP-Q01] Loading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)
    seq_len = 256
    n_blocks = min(n_batches, ids.numel() // seq_len)

    blocks = model.transformer.h
    n_layer = len(blocks)
    n_head = model.config.n_head
    stats = {(l, h): HeadStats(l, h) for l in range(n_layer) for h in range(n_head)}

    # per-position arrays to correlate head thermo with prediction entropy
    pos_heat, pos_pred_entropy, pos_svn = [], [], []
    pos_index = []
    # per-block scratch: heat/svn per layer (list of [Q] tensors over heads)
    block_heat = {l: [] for l in range(n_layer)}
    block_svn = {l: [] for l in range(n_layer)}

    print(f"[EXP-Q01] Processing {n_blocks} blocks of {seq_len} tokens...\n")
    with torch.no_grad():
        for b in range(n_blocks):
            block = ids[b * seq_len:(b + 1) * seq_len].unsqueeze(0).to(device)
            out = model(input_ids=block, output_hidden_states=True)
            hs = out.hidden_states                            # tuple len n_layer+1
            logits_lm = out.logits[0]                         # [T, vocab]
            pred_probs = logits_lm.softmax(dim=-1)
            pred_ent = -(pred_probs * pred_probs.clamp(min=1e-12).log()).sum(-1)  # [T]

            for l, blk in enumerate(blocks):
                hidden = hs[l]                                # input to layer l
                attn = blk.attn
                s, v = _attn_logits_and_values(model, hidden, l, attn)   # [1,H,T,T],[1,H,T,dh]
                s = s[0]; v = v[0]                            # [H,T,T], [H,T,dh]
                T = s.shape[1]
                # only query positions with enough context
                q_idx = list(range(min_context, T))
                if not q_idx:
                    continue
                for h in range(n_head):
                    sh = s[h, q_idx, :]                       # [Q, T] (with -inf above diag)
                    # restrict to valid (finite) keys per row handled by softmax of -inf
                    th = thermo_from_logits(sh)               # each [Q]
                    A = sh.softmax(dim=-1)                    # [Q, T]
                    Vh = v[h].unsqueeze(0).expand(len(q_idx), -1, -1)   # [Q, T, dh]
                    purity, svn = density_matrix_metrics(A, Vh)         # [Q]
                    agg = {
                        "free_energy": th["free_energy"].mean().item(),
                        "mean_E": th["mean_E"].mean().item(),
                        "heat_capacity": th["heat_capacity"].mean().item(),
                        "shannon_entropy": th["shannon_entropy"].mean().item(),
                        "T_eff": th["T_eff"].mean().item(),
                        "purity": purity.mean().item(),
                        "S_vn": svn.mean().item(),
                    }
                    stats[(l, h)].add(agg, len(q_idx))
                    # Accumulate per-position thermo, averaged over ALL heads/layers
                    # (not one frozen head). Normalize C per layer first so deep-layer
                    # logit growth doesn't dominate the correlation (confound control).
                    c_layer = th["heat_capacity"]
                    c_norm = c_layer / max(c_layer.mean().item(), 1e-9)
                    block_heat[l].append(c_norm)
                    block_svn[l].append(svn)
            # after all layers/heads for this block: aggregate per position
            # mean over layers and heads → one scalar per query position
            heat_stack = torch.stack([torch.stack(block_heat[l]).mean(0)
                                      for l in range(n_layer)]).mean(0)  # [Q]
            svn_stack = torch.stack([torch.stack(block_svn[l]).mean(0)
                                     for l in range(n_layer)]).mean(0)   # [Q]
            pos_heat.extend(heat_stack.tolist())
            pos_svn.extend(svn_stack.tolist())
            pos_pred_entropy.extend(pred_ent[q_idx].tolist())
            pos_index.extend(list(q_idx))    # sequence position, for confound control
            for l in range(n_layer):
                block_heat[l].clear(); block_svn[l].clear()
            print(f"  block {b+1}/{n_blocks} done")

    reports = [stats[k].finalize() for k in sorted(stats)]

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*78}\n[EXP-Q01] Per-head thermodynamics (sample)\n{'='*78}")
    for r in reports[:8] + reports[-8:]:
        print(" ", r)

    def corr(x, y):
        n = len(x)
        if n < 2:
            return 0.0
        mx, my = sum(x) / n, sum(y) / n
        cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
        sx = math.sqrt(sum((a - mx) ** 2 for a in x))
        sy = math.sqrt(sum((b - my) ** 2 for b in y))
        return cov / max(sx * sy, 1e-12)

    c_heat = corr(pos_heat, pos_pred_entropy)
    c_svn = corr(pos_svn, pos_pred_entropy)

    # Partial correlation controlling for sequence position (the obvious confound:
    # later tokens have more context → lower entropy AND more concentrated attention).
    # Lesson from NQP-U1b: a bivariate correlation can be entirely a confound.
    def partial_corr(x, y, z):
        def resid(a, ctrl):
            n = len(a); mc = sum(ctrl) / n; ma = sum(a) / n
            b = sum((c - mc) * (v - ma) for c, v in zip(ctrl, a)) / \
                max(sum((c - mc) ** 2 for c in ctrl), 1e-12)
            a0 = ma - b * mc
            return [v - (a0 + b * c) for c, v in zip(ctrl, a)]
        return corr(resid(x, z), resid(y, z))

    logpos = [math.log(p + 1) for p in pos_index]
    c_svn_partial = partial_corr(pos_svn, pos_pred_entropy, logpos)
    c_heat_partial = partial_corr(pos_heat, pos_pred_entropy, logpos)

    # spread of regimes across heads
    caps = [r.heat_capacity for r in reports]
    purs = [r.purity for r in reports]
    print(f"\n{'='*78}\n[EXP-Q01 VERDICT]\n{'='*78}")
    print(f"  heads analyzed: {len(reports)}  ({n_layer} layers × {n_head} heads)")
    print(f"  heat capacity C  range: [{min(caps):.3f}, {max(caps):.3f}]  "
          f"(spread ⇒ distinct thermodynamic regimes)")
    print(f"  purity Tr(ρ²)    range: [{min(purs):.3f}, {max(purs):.3f}]")
    print(f"  corr( heat_capacity , pred entropy ): {c_heat:+.3f}  "
          f"| partial (|pos): {c_heat_partial:+.3f}")
    print(f"  corr( S_vn(ρ)       , pred entropy ): {c_svn:+.3f}  "
          f"| partial (|pos): {c_svn_partial:+.3f}")
    # Decisive: does the correlation SURVIVE controlling for position?
    survives = abs(c_svn_partial) > 0.15 or abs(c_heat_partial) > 0.15
    if survives:
        print(f"  => SIGNAL SURVIVES position control: a thermodynamic quantity genuinely "
              f"tracks prediction uncertainty (not a position artifact). Pursue in Fase 2.")
    elif abs(c_svn) > 0.15 or abs(c_heat) > 0.15:
        print(f"  => CONFOUND: bivariate correlation collapses when controlling for position. "
              f"The signal was a sequence-position artifact, not thermodynamics (cf. NQP-U1b).")
    else:
        print(f"  => WEAK; check per-head regime structure instead of per-position.")

    return {"reports": reports, "corr_heat_predent": c_heat,
            "corr_svn_predent": c_svn, "n_heads": len(reports)}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="NQP-Q EXP-Q01 — attention thermodynamics")
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    run_exp_q01(n_batches=args.n_batches, device=args.device)
