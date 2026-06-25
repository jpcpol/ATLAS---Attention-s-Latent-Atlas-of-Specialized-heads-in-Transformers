"""
NQP-Q — Crystallization scaling law (EXP-Q04, GAP-M).

Question (J.P. Chancay): is the crystallization depth L_c universal or contingent
on model size? Measure the per-layer thermodynamic profile (R = p₂/p₁, argmax
stability, S_vn) — the cheap Q01 signature, forward-only — across GPT-2 sizes and
locate L_c (first layer whose mean R drops below a threshold). The trend across
{small, medium, large} discriminates the scenarios:

  1. L_c grows with depth        4. effect vanishes in large models
  2. L_c fixed (RG fixed point)  3. partial recrystallization

We report L_c both absolute and as a FRACTION of depth (L_c / N_layers), since a
constant fraction vs constant absolute value distinguishes scenario 1 from 2.
"""

from __future__ import annotations

import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch


def crystallization_profile(model_name: str, n_blocks=6, device="cpu",
                            noise_eps=0.05, min_context=16, seed=42):
    """Per-layer mean R=p₂/p₁ and argmax stability for one GPT-2 model."""
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset
    import os
    torch.manual_seed(seed)
    if str(device) == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    tok = GPT2TokenizerFast.from_pretrained(model_name); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name); model.eval()
    n_layer = model.config.n_layer
    n_head = model.config.n_head

    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)
    seq_len = 256
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    R_acc = [0.0] * n_layer
    stab_acc = [0.0] * n_layer
    cnt = [0] * n_layer

    nb = min(n_blocks, ids.numel() // seq_len)
    with torch.no_grad():
        for b in range(nb):
            blk = ids[b*seq_len:(b+1)*seq_len].unsqueeze(0).to(device)
            out = model(input_ids=blk, output_hidden_states=True)
            for l, layer in enumerate(model.transformer.h):
                hidden = out.hidden_states[l]
                attn = layer.attn
                B, T, D = hidden.shape
                dh = D // n_head
                qkv = attn.c_attn(hidden); q, k, _ = qkv.split(D, dim=2)
                shp = lambda x: x.view(B, T, n_head, dh).permute(0, 2, 1, 3)
                q, k = shp(q), shp(k)
                scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
                scores = scores.masked_fill(~mask, float("-inf"))
                qn = q + noise_eps * q.std() * torch.randn_like(q)
                scores_n = ((qn @ k.transpose(-1, -2)) / math.sqrt(dh)).masked_fill(~mask, float("-inf"))
                s = scores[0, :, min_context:, :]
                sn = scores_n[0, :, min_context:, :]
                p = s.softmax(-1)
                top2 = p.topk(2, dim=-1).values
                R = (top2[..., 1] / top2[..., 0].clamp(min=1e-9)).mean().item()
                stab = (s.argmax(-1) == sn.argmax(-1)).float().mean().item()
                R_acc[l] += R; stab_acc[l] += stab; cnt[l] += 1
            print(f"    [{model_name}] block {b+1}/{nb}", flush=True)

    Rs = [R_acc[l] / max(cnt[l], 1) for l in range(n_layer)]
    stabs = [stab_acc[l] / max(cnt[l], 1) for l in range(n_layer)]
    return {"model": model_name, "n_layer": n_layer, "R": Rs, "stab": stabs}


def find_Lc(R, thresh=0.05):
    """First layer index whose R stays below `thresh` for the rest of the network."""
    n = len(R)
    for l in range(n):
        if all(R[j] < thresh for j in range(l, n)):
            return l
    return n  # never fully crystallizes


def residual_intrinsic_dim(model_name, device="cpu", n_blocks=10, seed=42,
                           n_points=1500, n_heads_sample=8):
    """
    Intrinsic dimension (TwoNN) of the attention residual ε, with a CONTROLLED
    protocol for cross-model comparability (J.P. Chancay): identical #points per
    head (n_points), identical #heads sampled (n_heads_sample), same depth fraction
    (last 3 layers — same ABSOLUTE count, not "third", so point composition matches).

    Returns (mean_int, std_int, mean_lin, per_head_ints) — std exposes whether the
    "≈const" mean hides spread (artifact check).
    """
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset
    from intrinsic import twonn_dimension, linear_dim_90, collect_residuals
    import os, statistics
    torch.manual_seed(seed)
    if str(device) == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    tok = GPT2TokenizerFast.from_pretrained(model_name); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name); model.eval()
    n_layer = model.config.n_layer
    # SAME absolute depth window across models (last 3 layers) for matched composition
    deep = tuple(range(n_layer - 3, n_layer))
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)
    res = collect_residuals(model, ids, 256, device, deep, n_blocks=n_blocks,
                            max_points=n_points + 500)
    keys = sorted(res.keys())
    # Use the SAME N across all heads/models = min available, capped at n_points.
    avail = min(res[k].shape[0] for k in keys)
    N = min(avail, n_points)
    ints, lins = [], []
    for key in keys[:n_heads_sample]:
        E = res[key][:N]                              # EXACT same N per head & model
        ints.append(twonn_dimension(E)); lins.append(linear_dim_90(E))
    print(f"    (controlled: N={N} pts/head, {len(ints)} heads, layers {deep})")
    return (statistics.mean(ints), statistics.pstdev(ints),
            statistics.mean(lins), ints)


def run_exp_q04(models=("gpt2", "gpt2-medium"), n_blocks=6, device="cpu", thresh=0.05,
                with_intrinsic=True):
    """
    EXP-Q04-lite — crystallization depth L_c AND residual intrinsic dim across sizes.
    Tests both: (a) does crystallization scale? (b) is dim(M_ε) ≈ const?
    """
    results = []
    for m in models:
        print(f"\n[EXP-Q04] Profiling {m}...")
        prof = crystallization_profile(m, n_blocks=n_blocks, device=device)
        Lc = find_Lc(prof["R"], thresh)
        prof["Lc"] = Lc
        prof["Lc_frac"] = Lc / prof["n_layer"]
        if with_intrinsic:
            print(f"  measuring residual intrinsic dim for {m} (controlled protocol)...")
            dint, dstd, dlin, per_head = residual_intrinsic_dim(m, device=device, n_blocks=10)
            prof["dim_int"] = dint; prof["dim_int_std"] = dstd; prof["dim_lin"] = dlin
            prof["per_head"] = per_head
        results.append(prof)
        print(f"  {m}: n_layer={prof['n_layer']}  L_c={Lc} (frac={prof['Lc_frac']:.2f})"
              + (f"  dim_int={prof['dim_int']:.1f}±{prof['dim_int_std']:.1f}  "
                 f"dim_lin={prof['dim_lin']:.1f}  (per-head: "
                 f"{[f'{d:.1f}' for d in prof['per_head']]})"
                 if with_intrinsic else ""))
        print("   R by layer: " + " ".join(f"{r:.2f}" for r in prof["R"]))

    print(f"\n{'='*72}\n[EXP-Q04 VERDICT] crystallization & effective-dim scaling\n{'='*72}")
    has_int = all("dim_int" in r for r in results)
    hdr = f"  {'model':>14} | {'n_layer':>7} | {'L_c':>4} | {'L_c/N':>6}"
    if has_int:
        hdr += f" | {'dim_int':>7} | {'dim_lin':>7}"
    print(hdr)
    for r in results:
        line = f"  {r['model']:>14} | {r['n_layer']:>7} | {r['Lc']:>4} | {r['Lc_frac']:>6.2f}"
        if has_int:
            line += f" | {r['dim_int']:>7.1f} | {r['dim_lin']:>7.1f}"
        print(line)

    if len(results) >= 2:
        fracs = [r["Lc_frac"] for r in results]
        abss = [r["Lc"] for r in results]
        if max(abss) - min(abss) <= 1:
            print(f"  L_c => SCENARIO 2 (fixed point): L_c ≈ const absolute ({abss}) — RG-like.")
        elif max(fracs) - min(fracs) < 0.10:
            print(f"  L_c => SCENARIO 1 (scales): L_c/N ≈ const ({[f'{f:.2f}' for f in fracs]}).")
        else:
            print(f"  L_c => MIXED: inspect per-layer R (recrystallization/vanishing).")

        if has_int:
            dints = [r["dim_int"] for r in results]
            stds = [r["dim_int_std"] for r in results]
            spread = max(dints) - min(dints)
            mean_within = sum(stds) / len(stds)        # avg within-model head spread
            print(f"\n  *** KEY HYPOTHESIS dim(M_ε) ≈ const ***")
            print(f"  intrinsic dims across sizes: "
                  f"{[f'{d:.1f}±{s:.1f}' for d, s in zip(dints, stds)]}")
            print(f"  between-model spread = {spread:.1f}  |  within-model head std ≈ {mean_within:.1f}")
            # Constant only if between-model spread is small RELATIVE to within-model noise
            if spread < 3.0 and spread <= 1.5 * mean_within:
                print(f"  => CONSTANT EFFECTIVE DIM (robust): between-model variation ({spread:.1f}) "
                      f"is within head-level noise ({mean_within:.1f}). ~{sum(dints)/len(dints):.0f}D "
                      f"manifold regardless of scale — effective-DOF reduction. Prioritize Q06.")
            elif spread < 3.0:
                print(f"  => CONSTANT-ish but between-model spread ({spread:.1f}) exceeds head noise "
                      f"({mean_within:.1f}); suggestive, not conclusive. Need more sizes/samples.")
            else:
                print(f"  => dim_int grows with size (spread {spread:.1f}); scale-dependent.")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="NQP-Q EXP-Q04 — crystallization scaling law")
    p.add_argument("--models", type=str, nargs="+", default=["gpt2", "gpt2-medium"])
    p.add_argument("--n-blocks", type=int, default=6)
    p.add_argument("--thresh", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    run_exp_q04(models=tuple(args.models), n_blocks=args.n_blocks,
                thresh=args.thresh, device=args.device)
