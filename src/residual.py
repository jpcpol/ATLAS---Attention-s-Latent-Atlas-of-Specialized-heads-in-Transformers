"""
NQP-Q — Crystallized attention + residual excitations (EXP-Q03, GAP-L).

Q02 showed hard Top-1 (Attn ≈ V_{i*}) breaks the model: the residual
ε = Σ_{i≠i*} p_i V_i is NOT negligible. GAP-L (J.P. Chancay) asks the better
question: is ε COMPRESSIBLE? Decomposition is exact —

    Attn = V_{i*} + ε ,    ε = Σ_{i≠i*} p_i V_i

Physical reading: ground state + excitations (|ψ⟩ = |0⟩ + ε|1⟩ + …), NOT a+ib.
ε is sub-dominant, so the analogy is ground-state+excitations, not real/imaginary.

Three replacement models on the deep (crystallized) layers:
  A: Top-1 pure          out = V_{i*}                         (Q02 baseline, breaks)
  B: Top-1 + full ε      out = V_{i*} + ε  == full softmax    (exactness sanity)
  C: Top-1 + low-rank ε  out = V_{i*} + P_r P_rᵀ ε            (the real test)

P_r is the top-d_r PCA basis of the per-head value vectors (calibration), i.e. the
optimal d_r-dim subspace to represent the excitations without retraining. We sweep
d_r and trace ΔPPL(d_r): if small d_r recovers most of the lost PPL, deep layers are
"nearly crystalline with few excitations carrying the needed information."
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
from torch import Tensor

from crystallize import perplexity


# ---------------------------------------------------------------------------
# Per-head value PCA basis (the subspace excitations are compressed into)
# ---------------------------------------------------------------------------

def collect_residual_svd(model, ids, seq_len, device, layers, n_blocks=4):
    """
    Collect the EXCITATION RESIDUALS ε = Attn - V_{i*} per (layer, head) across many
    positions, then SVD them. The basis we compress into is the top-r left-singular
    vectors of the residuals THEMSELVES — the correct question (J.P. Chancay) is
    whether ε lives in a low-dim subspace, not whether the values do.

    Returns:
      basis: (layer,head) -> U_r [d_head, d_head] (left singular vecs, descending)
      spectrum: (layer,head) -> cumulative variance ratio per rank (effective dim)
    """
    n_head = model.config.n_head
    bufs = {}
    handles = []

    def make_hook(li):
        def hook(module, inp, out):
            hidden = inp[0] if isinstance(inp, tuple) else inp
            B, T, D = hidden.shape
            dh = D // n_head
            qkv = module.c_attn(hidden)
            q, k, v = qkv.split(D, dim=2)
            shp = lambda x: x.view(B, T, n_head, dh).permute(0, 2, 1, 3)
            q, k, v = shp(q), shp(k), shp(v)
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            mask = torch.tril(torch.ones(T, T, device=hidden.device, dtype=torch.bool))
            scores = scores.masked_fill(~mask, float("-inf"))
            p = scores.softmax(dim=-1)
            ctx = p @ v
            i_star = p.argmax(dim=-1)
            v_star = torch.gather(v, 2, i_star.unsqueeze(-1).expand(-1, -1, -1, dh))
            eps = (ctx - v_star)[0]                              # [H, T, dh]
            for h in range(n_head):
                bufs.setdefault((li, h), []).append(eps[h].detach())
        return hook

    for li in layers:
        handles.append(model.transformer.h[li].attn.register_forward_hook(make_hook(li)))

    nb = min(n_blocks, ids.numel() // seq_len)
    with torch.no_grad():
        for b in range(nb):
            blk = ids[b*seq_len:(b+1)*seq_len].unsqueeze(0).to(device)
            model(input_ids=blk)
    for hd in handles:
        hd.remove()

    basis, spectrum = {}, {}
    for key, chunks in bufs.items():
        E = torch.cat(chunks, dim=0)                            # [N, dh] residuals
        # SVD of the residual matrix: right singular vectors = residual modes
        _, S, Vt = torch.linalg.svd(E, full_matrices=False)
        basis[key] = Vt.t()                                     # [dh, dh] modes (descending)
        var = (S ** 2)
        spectrum[key] = (var.cumsum(0) / var.sum().clamp(min=1e-12))
    return basis, spectrum


# ---------------------------------------------------------------------------
# Patched attention: V_{i*} + low-rank residual
# ---------------------------------------------------------------------------

def make_residual_attn(orig_module, layer_idx, basis, d_r, mode="lowrank"):
    """
    mode: 'top1' (V_{i*}), 'full' (exact), 'lowrank' (V_{i*} + P_r P_rᵀ ε).
    basis: dict (layer,head)->[dh,dh] PCA eigenvectors.
    """
    n_head = orig_module.num_heads if hasattr(orig_module, "num_heads") else orig_module.n_head

    def forward(hidden_states, **kwargs):
        B, T, D = hidden_states.shape
        dh = D // n_head
        qkv = orig_module.c_attn(hidden_states)
        q, k, v = qkv.split(D, dim=2)
        shp = lambda x: x.view(B, T, n_head, dh).permute(0, 2, 1, 3)
        q, k, v = shp(q), shp(k), shp(v)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        mask = torch.tril(torch.ones(T, T, device=hidden_states.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        p = scores.softmax(dim=-1)                              # [B,H,T,T] full weights

        # full attention output and its Top-1 (ground-state) component
        ctx_full = p @ v                                        # [B,H,T,dh]
        i_star = p.argmax(dim=-1)                               # [B,H,T]
        v_star = torch.gather(v, 2, i_star.unsqueeze(-1).expand(-1, -1, -1, dh))  # [B,H,T,dh]

        if mode == "top1":
            ctx = v_star
        elif mode == "full":
            ctx = ctx_full
        else:  # lowrank: V_{i*} + P_r P_rᵀ (ctx_full - V_{i*})
            eps = ctx_full - v_star                            # [B,H,T,dh] residual
            outs = []
            for h in range(n_head):
                Pr = basis[(layer_idx, h)][:, :d_r].to(v.device)   # [dh, d_r]
                proj = (eps[:, h] @ Pr) @ Pr.t()               # [B,T,dh]
                outs.append(v_star[:, h] + proj)
            ctx = torch.stack(outs, dim=1)                     # [B,H,T,dh]

        ctx = ctx.permute(0, 2, 1, 3).reshape(B, T, D)
        return (orig_module.c_proj(ctx), None)

    return forward


class ResidualPatch:
    def __init__(self, model, layers, basis, d_r, mode):
        self.model, self.layers, self.basis = model, layers, basis
        self.d_r, self.mode, self._orig = d_r, mode, {}

    def __enter__(self):
        for li in self.layers:
            attn = self.model.transformer.h[li].attn
            self._orig[li] = attn.forward
            attn.forward = make_residual_attn(attn, li, self.basis, self.d_r, self.mode)
        return self

    def __exit__(self, *a):
        for li, fwd in self._orig.items():
            self.model.transformer.h[li].attn.forward = fwd
        self._orig.clear()


# ---------------------------------------------------------------------------
# EXP-Q03
# ---------------------------------------------------------------------------

def run_exp_q03(
    deep_layers=(9, 10, 11),
    d_r_sweep=(1, 2, 4, 8, 16, 32),
    n_blocks: int = 30,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    EXP-Q03 — does a low-rank residual recover the PPL that hard Top-1 destroys?
    Crystallize `deep_layers`, compare A/B/C and sweep the residual rank d_r.
    """
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset
    import os
    torch.manual_seed(seed)
    if str(device) == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    print("[EXP-Q03] Loading GPT-2 small...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2"); model.eval()
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)
    seq_len = 256
    d_head = model.config.n_embd // model.config.n_head

    ppl_fp = perplexity(model, ids, seq_len, device, n_blocks)
    print(f"  PPL baseline (full softmax)       = {ppl_fp:.3f}")

    print(f"[EXP-Q03] SVD of excitation residuals ε for layers {deep_layers}...")
    basis, spectrum = collect_residual_svd(model, ids, seq_len, device, deep_layers, n_blocks=4)

    # Effective dimension of the residual: how many SVD modes for X% variance
    print(f"\n  Residual effective dimension (cumulative variance, d_head={d_head}):")
    print(f"  {'layer/head':>12} | r=1   r=2   r=4   r=8   r=16")
    eff_dims = []
    for key in sorted(spectrum):
        cv = spectrum[key]
        getr = lambda r: cv[min(r-1, len(cv)-1)].item()
        # effective dim = smallest r reaching 90% variance
        r90 = int((cv < 0.90).sum().item()) + 1
        eff_dims.append(r90)
        if key[1] == 0:  # print head 0 of each layer to keep it short
            print(f"  L{key[0]:2d}H{key[1]:2d}      | "
                  f"{getr(1):.2f}  {getr(2):.2f}  {getr(4):.2f}  {getr(8):.2f}  {getr(16):.2f}")
    import statistics
    print(f"  mean effective dim (r for 90% var): {statistics.mean(eff_dims):.1f} / {d_head}")

    # Model A: Top-1 pure
    with ResidualPatch(model, deep_layers, basis, 0, "top1"):
        ppl_a = perplexity(model, ids, seq_len, device, n_blocks)
    # Model B: full residual (exactness sanity — should equal baseline)
    with ResidualPatch(model, deep_layers, basis, d_head, "full"):
        ppl_b = perplexity(model, ids, seq_len, device, n_blocks)
    print(f"  [A] Top-1 pure (V_i*)             = {ppl_a:.3f}  ΔPPL={ppl_a-ppl_fp:+.3f}")
    print(f"  [B] Top-1 + full ε (sanity)       = {ppl_b:.3f}  ΔPPL={ppl_b-ppl_fp:+.3f}  "
          f"(should ≈ 0)")

    # Model C: low-rank residual sweep
    print(f"\n  [C] Top-1 + low-rank residual (d_head={d_head}):")
    curve = []
    for d_r in d_r_sweep:
        with ResidualPatch(model, deep_layers, basis, d_r, "lowrank"):
            ppl_c = perplexity(model, ids, seq_len, device, n_blocks)
        recovered = (ppl_a - ppl_c) / max(ppl_a - ppl_fp, 1e-9) * 100
        curve.append((d_r, ppl_c, ppl_c - ppl_fp, recovered))
        print(f"      d_r={d_r:2d}/{d_head}: PPL={ppl_c:8.3f}  ΔPPL={ppl_c-ppl_fp:+7.3f}  "
              f"recovered={recovered:5.1f}% of Top-1 loss")

    print(f"\n{'='*66}\n[EXP-Q03 VERDICT] is the excitation residual compressible?\n{'='*66}")
    # smallest d_r recovering ≥90% of the loss
    good = [(dr, rec) for (dr, _, _, rec) in curve if rec >= 90.0]
    if good:
        dr_min = min(d for d, _ in good)
        print(f"  => YES: d_r={dr_min}/{d_head} recovers ≥90% of the Top-1 loss. Deep layers are "
              f"nearly crystalline + a rank-{dr_min} excitation. Compressible attention found.")
    else:
        best = max(curve, key=lambda c: c[3])
        print(f"  => PARTIAL: best is d_r={best[0]} recovering {best[3]:.0f}%. The residual is "
              f"not strongly low-rank; excitations are spread across many value dimensions.")
    return {"ppl_fp": ppl_fp, "ppl_top1": ppl_a, "ppl_full": ppl_b, "curve": curve}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="NQP-Q EXP-Q03 — residual excitation compression")
    p.add_argument("--n-blocks", type=int, default=30)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--layers", type=int, nargs="+", default=[9, 10, 11])
    args = p.parse_args()
    run_exp_q03(deep_layers=tuple(args.layers), n_blocks=args.n_blocks, device=args.device)
