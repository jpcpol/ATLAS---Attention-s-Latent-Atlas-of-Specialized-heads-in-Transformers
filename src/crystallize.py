"""
NQP-Q — Attention Crystallization (EXP-Q02).

Hypothesis (J.P. Chancay, 2026-06-25): heads with T_eff→1 (S_vn→0) have collapsed
to a near-delta softmax, so Attn ≈ V_{argmax}. In that frozen regime the exp/sum/
division of softmax is wasted compute — replace with Top-k selection. If deep layers
are crystallized (EXP-Q01: L11 has purity=1.0, T_eff=1), replacing their attention
with Top-1/Top-2 should barely move PPL while cutting compute and memory.

  E1: per-head determinism R = p₂/p₁; ablate each head→argmax; measure ΔPPL.
  E2: progressive layer replacement L11→argmax, L10-11, …; curve ΔPPL(k) reveals
      the network's "freezing point" (liquid → critical → frozen phases).

Trap (also from J.P.): S_vn≈0 does NOT imply replaceable — the model may sit near a
transition where ∂Attn/∂q still matters. So we MEASURE ablation impact, never assume.

Implementation: monkey-patch GPT2Attention's softmax with a Top-k hard selection on
chosen layers, run on WikiText-103, compare PPL to the FP baseline.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Top-k hard attention (the crystallized replacement)
# ---------------------------------------------------------------------------

def topk_attention_weights(scores: Tensor, k: int) -> Tensor:
    """
    Replace softmax with a hard Top-k selection (renormalized over the k kept).

    scores: [..., n_keys] (causal-masked with -inf above diagonal).
    k=1 → pure argmax (output = V_{argmax}); k=2 → top-2 renormalized; etc.

    For k=1 this is the fully crystallized limit: no exp, just a one-hot gather.
    We keep the kept-weights' RELATIVE softmax so k>1 stays a valid mixture, which
    is cheaper than full softmax (k≪T) and matches the "near-delta" regime.
    """
    n = scores.shape[-1]
    kk = min(k, n)
    topv, topi = scores.topk(kk, dim=-1)                  # [..., k]
    # softmax only over the k kept logits (relative), zeros elsewhere
    w_kept = topv.softmax(dim=-1)                         # [..., k]
    out = torch.zeros_like(scores)
    out.scatter_(-1, topi, w_kept)
    return out


def make_topk_gpt2_attn(orig_module, k: int):
    """
    Build a forward that mimics GPT2Attention but uses Top-k weights instead of
    softmax. Reproduces GPT-2's Conv1D q/k/v split and causal mask.
    """
    n_head = orig_module.num_heads if hasattr(orig_module, "num_heads") else orig_module.n_head

    def forward(hidden_states, **kwargs):
        B, T, D = hidden_states.shape
        qkv = orig_module.c_attn(hidden_states)
        q, k_, v = qkv.split(D, dim=2)
        dh = D // n_head

        def shape(x):
            return x.view(B, T, n_head, dh).permute(0, 2, 1, 3)

        q, k_, v = shape(q), shape(k_), shape(v)
        scores = (q @ k_.transpose(-1, -2)) / math.sqrt(dh)      # [B,H,T,T]
        mask = torch.tril(torch.ones(T, T, device=hidden_states.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))

        w = topk_attention_weights(scores, k)                    # [B,H,T,T] hard
        ctx = w @ v                                              # [B,H,T,dh]
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, T, D)
        out = orig_module.c_proj(ctx)
        # GPT2 returns (attn_output, present, (attentions)) — match minimal shape
        return (out, None)

    return forward


class CrystallizePatch:
    """Context manager: temporarily replace attention forward on chosen layers."""
    def __init__(self, model, layer_indices, k: int):
        self.model = model
        self.layers = layer_indices
        self.k = k
        self._orig = {}

    def __enter__(self):
        for li in self.layers:
            attn = self.model.transformer.h[li].attn
            self._orig[li] = attn.forward
            attn.forward = make_topk_gpt2_attn(attn, self.k)
        return self

    def __exit__(self, *a):
        for li, fwd in self._orig.items():
            self.model.transformer.h[li].attn.forward = fwd
        self._orig.clear()


# ---------------------------------------------------------------------------
# Phase A — per-head crystallization diagnostics (controls C1, C2)
# ---------------------------------------------------------------------------

@dataclass
class CrystalDiag:
    layer: int
    head: int
    margin: float        # C1: p₁ - p₂ (1 = fully crystallized)
    ratio: float         # C1: R = p₂/p₁ (≪1 = genuinely deterministic)
    argmax_stable: float # C2: P(argmax unchanged under q+ε noise)

    def __repr__(self):
        return (f"L{self.layer:2d}H{self.head:2d}  "
                f"margin(p₁-p₂)={self.margin:.3f}  R=p₂/p₁={self.ratio:.3f}  "
                f"argmax_stable={self.argmax_stable:.3f}")


def run_phase_a(n_blocks: int = 8, device: str = "cpu", seed: int = 42,
                noise_eps: float = 0.05, min_context: int = 16) -> list:
    """
    Phase A — measure per-head determinism (C1: margin/ratio) and robustness
    (C2: argmax stability under input noise q+ε). A head is a safe crystallization
    candidate only if R≪1 AND argmax is stable; low S_vn alone is insufficient.
    """
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset
    import os
    torch.manual_seed(seed)
    if str(device) == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2"); model.eval()
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)
    seq_len = 256
    n_layer = len(model.transformer.h); n_head = model.config.n_head

    acc = {(l, h): [0.0, 0.0, 0.0, 0] for l in range(n_layer) for h in range(n_head)}
    nb = min(n_blocks, ids.numel() // seq_len)
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    with torch.no_grad():
        for b in range(nb):
            blk = ids[b*seq_len:(b+1)*seq_len].unsqueeze(0).to(device)
            out = model(input_ids=blk, output_hidden_states=True)
            for l, layer in enumerate(model.transformer.h):
                hidden = out.hidden_states[l]
                attn = layer.attn
                B, T, D = hidden.shape
                dh = D // n_head
                qkv = attn.c_attn(hidden); q, k_, _ = qkv.split(D, dim=2)
                sh = lambda x: x.view(B, T, n_head, dh).permute(0, 2, 1, 3)
                q, k_ = sh(q), sh(k_)                          # [1,H,T,dh]
                scores = (q @ k_.transpose(-1, -2)) / math.sqrt(dh)
                scores = scores.masked_fill(~mask, float("-inf"))
                qn = q + noise_eps * q.std() * torch.randn_like(q)
                scores_n = ((qn @ k_.transpose(-1, -2)) / math.sqrt(dh)).masked_fill(~mask, float("-inf"))
                # Vectorized over ALL heads at once: [H, Q, T]
                s = scores[0, :, min_context:, :]
                sn = scores_n[0, :, min_context:, :]
                p = s.softmax(-1)
                top2 = p.topk(2, dim=-1).values               # [H,Q,2]
                p1, p2 = top2[..., 0], top2[..., 1]
                margin = (p1 - p2).mean(dim=-1)                # [H]
                ratio = (p2 / p1.clamp(min=1e-9)).mean(dim=-1) # [H]
                stable = (s.argmax(-1) == sn.argmax(-1)).float().mean(dim=-1)  # [H]
                for h in range(n_head):
                    a = acc[(l, h)]
                    a[0] += margin[h].item(); a[1] += ratio[h].item()
                    a[2] += stable[h].item(); a[3] += 1
            print(f"  [Phase A] block {b+1}/{nb} done", flush=True)

    diags = []
    for (l, h), a in sorted(acc.items()):
        n = max(a[3], 1)
        diags.append(CrystalDiag(l, h, a[0]/n, a[1]/n, a[2]/n))
    return diags


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def perplexity(model, ids: Tensor, seq_len: int, device, max_blocks: int) -> float:
    n_blocks = min(max_blocks, ids.numel() // seq_len)
    tot_nll, tot_tok = 0.0, 0
    with torch.no_grad():
        for b in range(n_blocks):
            blk = ids[b * seq_len:(b + 1) * seq_len].unsqueeze(0).to(device)
            loss = model(input_ids=blk, labels=blk).loss.item()
            if math.isfinite(loss):
                tot_nll += loss * (seq_len - 1)
                tot_tok += seq_len - 1
    return math.exp(tot_nll / max(tot_tok, 1))


def run_exp_q02(
    n_blocks: int = 40,
    device: str = "cpu",
    seed: int = 42,
    ks=(1, 2),
) -> dict:
    """
    EXP-Q02 — progressive crystallization (E2). Replace attention with Top-k in the
    deepest layers, growing the frozen set, and trace ΔPPL vs depth threshold.
    """
    try:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("requires transformers, datasets, torch") from e

    torch.manual_seed(seed)
    import os
    if str(device) == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    print("[EXP-Q02] Loading GPT-2 small...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    print("[EXP-Q02] Loading WikiText-103 validation...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)
    seq_len = 256
    n_layer = len(model.transformer.h)

    ppl_fp = perplexity(model, ids, seq_len, device, n_blocks)
    print(f"\n  PPL baseline (full softmax) = {ppl_fp:.3f}\n")

    # ── Phase A: per-layer determinism (C1 ratio R, C2 argmax stability) ──────
    print(f"{'='*64}\n[Phase A] Per-layer determinism & robustness\n{'='*64}")
    diags = run_phase_a(n_blocks=min(n_blocks, 8), device=device, seed=seed)
    by_layer = {}
    for d in diags:
        by_layer.setdefault(d.layer, []).append(d)
    print("  layer | mean R=p₂/p₁ | mean argmax_stable | crystallized? (R<0.2 & stab>0.9)")
    layer_cryst = {}
    for l in sorted(by_layer):
        ds_l = by_layer[l]
        mR = sum(x.ratio for x in ds_l) / len(ds_l)
        mS = sum(x.argmax_stable for x in ds_l) / len(ds_l)
        cryst = mR < 0.2 and mS > 0.9
        layer_cryst[l] = cryst
        print(f"   L{l:2d}  |    {mR:.3f}     |       {mS:.3f}        |  {'YES' if cryst else 'no'}")
    results = {"ppl_fp": ppl_fp, "curves": {}, "phase_a": diags, "layer_cryst": layer_cryst}

    for k in ks:
        print(f"{'='*64}\n[E2] Progressive crystallization with Top-{k}\n{'='*64}")
        curve = []
        # threshold L: crystallize layers [L .. n_layer-1]; L=n_layer means none
        for L in range(n_layer, -1, -1):
            layers = list(range(L, n_layer))
            if layers:
                with CrystallizePatch(model, layers, k):
                    ppl = perplexity(model, ids, seq_len, device, n_blocks)
            else:
                ppl = ppl_fp
            n_cryst = len(layers)
            dppl = ppl - ppl_fp
            curve.append((n_cryst, ppl, dppl))
            print(f"  crystallized deepest {n_cryst:2d} layers (L{L}-{n_layer-1}): "
                  f"PPL={ppl:8.3f}  ΔPPL={dppl:+8.3f}")
        results["curves"][k] = curve

    # ── Verdict: how many deep layers can we crystallize under a PPL budget? ──
    print(f"\n{'='*64}\n[EXP-Q02 VERDICT] freezing point of GPT-2\n{'='*64}")
    for k in ks:
        curve = results["curves"][k]
        # largest #layers crystallized while ΔPPL stays under thresholds
        for budget in (0.5, 1.0, 5.0):
            ok = [n for (n, _, d) in curve if d <= budget and n > 0]
            best = max(ok) if ok else 0
            print(f"  Top-{k}: can crystallize {best:2d}/{n_layer} deepest layers "
                  f"within ΔPPL ≤ {budget}")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="NQP-Q EXP-Q02 — attention crystallization")
    p.add_argument("--n-blocks", type=int, default=40)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2])
    args = p.parse_args()
    run_exp_q02(n_blocks=args.n_blocks, device=args.device, ks=tuple(args.ks))
