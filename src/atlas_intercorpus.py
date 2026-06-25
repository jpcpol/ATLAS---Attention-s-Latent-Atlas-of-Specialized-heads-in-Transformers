"""
NQP-Q — Inter-corpus control for the inter-head overlap O_h (pre-submission).

The single most obvious reviewer question: "is non-alignment a property of the
MODEL, or of the corpus you measured it on?" We answer it by recomputing O_h on a
SECOND corpus (C4) and comparing against WikiText-103, with bootstrap CIs.

Expected outcomes (J.P. Chancay's framing):
  A  WT 0.28 / C4 0.26  → perfect, data-independent.
  B  WT 0.28 / C4 0.18  → still publishable: degree is data-dependent, EXISTENCE robust.
  C  WT 0.28 / C4 0.95  → essentially impossible given everything else measured.

Same fixed protocol as atlas_robustness.py (deepest layer, n_heads, N, d_local=7),
only the text source changes.
"""

from __future__ import annotations

import sys
import statistics

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch

from intrinsic import collect_residuals
from atlas_robustness import pairwise_overlaps, bootstrap_ci, deepest_layer_heads


def _corpus_ids(corpus, tok, max_chars=600_000):
    """Return a token-id 1-D tensor for the given corpus name."""
    from datasets import load_dataset
    if corpus == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    elif corpus == "c4":
        # C4 is huge → stream a handful of validation docs until we have enough text
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        parts, total = [], 0
        for ex in ds:
            t = ex["text"].strip()
            if not t:
                continue
            parts.append(t); total += len(t)
            if total >= max_chars:
                break
        text = "\n\n".join(parts)
    else:
        raise ValueError(corpus)
    return tok(text[:max_chars], return_tensors="pt")["input_ids"].squeeze(0)


def run(model_name="gpt2", corpora=("wikitext", "c4"), device="cpu",
        n_blocks=12, n_heads=8, n_points=1200, d_local=7, seed=42):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    import os
    torch.manual_seed(seed)
    if str(device) == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    print(f"[Inter-corpus] {model_name}  corpora={corpora}  (d_local={d_local})")
    tok = GPT2TokenizerFast.from_pretrained(model_name); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name); model.eval()
    deep_layer = model.config.n_layer - 1

    results = []
    for corpus in corpora:
        print(f"\n  [{corpus}] loading text + collecting residuals (layer {deep_layer})...")
        ids = _corpus_ids(corpus, tok)
        by_head, N = deepest_layer_heads(model, ids, deep_layer, device,
                                         n_blocks, n_points, n_heads)
        pair_vals = pairwise_overlaps(by_head, d_local)
        mean, lo, hi, sd = bootstrap_ci(pair_vals)
        results.append({"corpus": corpus, "O_h": mean, "ci": (lo, hi), "N": N})
        print(f"  [{corpus}] N={N}  O_h = {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    print(f"\n{'='*66}\n[Inter-corpus VERDICT] {model_name}\n{'='*66}")
    print(f"  {'corpus':>10} | {'O_h':>6} | {'95% CI':>16} | {'N':>5}")
    for r in results:
        print(f"  {r['corpus']:>10} | {r['O_h']:>6.3f} | "
              f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {r['N']:>5}")
    ohs = [r["O_h"] for r in results]
    spread = max(ohs) - min(ohs)
    all_low = all(o < 0.5 for o in ohs)
    # do the CIs overlap?
    los = [r["ci"][0] for r in results]; his = [r["ci"][1] for r in results]
    overlap_ci = max(los) <= min(his)
    print(f"\n  cross-corpus spread: {spread:.3f}   CIs overlap: {overlap_ci}")
    if all_low and overlap_ci:
        print(f"  => DATA-INDEPENDENT: O_h ≪ 1 on both corpora and CIs overlap. "
              f"Non-alignment is a property of the model, not the measurement corpus.")
    elif all_low:
        print(f"  => EXISTENCE ROBUST, degree data-dependent: O_h ≪ 1 on both corpora "
              f"(spread {spread:.2f}). The DEGREE of non-alignment shifts with data; its "
              f"EXISTENCE does not.")
    else:
        print(f"  => WARNING: O_h not ≪ 1 on some corpus — re-examine (unexpected).")
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="NQP-Q — inter-corpus control for O_h")
    p.add_argument("--model", type=str, default="gpt2")
    p.add_argument("--corpora", type=str, nargs="+", default=["wikitext", "c4"])
    p.add_argument("--n-blocks", type=int, default=12)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-points", type=int, default=1200)
    p.add_argument("--d-local", type=int, default=7)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    run(model_name=args.model, corpora=tuple(args.corpora), device=args.device,
        n_blocks=args.n_blocks, n_heads=args.n_heads, n_points=args.n_points,
        d_local=args.d_local)
