"""
Phase 0 regression gate — the refactor must NOT move the frozen GPT-2 numbers.

The cross-architecture work (Llama/Mistral) is built on a backend refactor of
`intrinsic.collect_residuals`. Gate P0 (see docs/cross_architecture_plan.md §5):
nothing ships unless GPT-2's O_h is reproduced exactly through the new code path.

This script:
  (A) GPT-2 regression — recomputes O_h on gpt2 small via the backend dispatch and
      checks it lands on the frozen value 0.284 (95% CI [0.276, 0.292]).
  (B) Llama/Mistral smoke (optional) — if --llama <model> is given, runs the
      LlamaBackend end-to-end and checks the residual clouds are sane (right shape,
      single-digit TwoNN per head), exercising RMSNorm+RoPE+GQA. No numeric target.

Run:
    python tests/test_phase0_regression.py                 # GPT-2 gate only
    python tests/test_phase0_regression.py --llama meta-llama/Llama-3.2-1B
"""

from __future__ import annotations

import os
import sys
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from atlas_robustness import _load, deepest_layer_heads, pairwise_overlaps, bootstrap_ci
from intrinsic import collect_residuals, twonn_dimension
from residual_backends import get_backend, model_head_geometry, GPT2Backend, LlamaBackend

# Frozen reference (src/atlas_robustness.py run, gpt2 small, deepest layer, N=1200, d_local=7)
FROZEN_OH = 0.284
FROZEN_CI = (0.276, 0.292)
TOL = 0.004                      # allow tiny drift from RNG/sched; well inside the CI width


def test_gpt2_regression(device="cpu"):
    print("=" * 70)
    print("[P0-A] GPT-2 regression: O_h via the new backend dispatch")
    print("=" * 70)

    model, ids = _load("gpt2", device, seed=42)

    # sanity: dispatch resolves to the GPT-2 backend
    be = get_backend(model)
    assert isinstance(be, GPT2Backend), f"expected GPT2Backend, got {type(be).__name__}"
    geo = model_head_geometry(model)
    assert geo == {"family": "gpt2", "n_layer": 12, "n_q": 12, "n_kv": 12,
                   "d_head": 64, "n_rep": 1}, geo
    print(f"  backend = GPT2Backend  geometry = {geo}")

    # same protocol as atlas_robustness.run: deepest layer, 8 heads, N=1200, d_local=7
    by_head, N = deepest_layer_heads(model, ids, layer=11, device=device,
                                     n_blocks=12, n_points=1200, n_heads=8)
    pair_vals = pairwise_overlaps(by_head, d_local=7)
    mean, lo, hi, sd = bootstrap_ci(pair_vals)
    print(f"  O_h = {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (N={N}, {len(pair_vals)} pairs)")
    print(f"  frozen target O_h = {FROZEN_OH}  CI {FROZEN_CI}")

    assert abs(mean - FROZEN_OH) <= TOL, (
        f"REGRESSION: O_h={mean:.3f} differs from frozen {FROZEN_OH} by >{TOL}")
    assert lo >= FROZEN_CI[0] - TOL and hi <= FROZEN_CI[1] + TOL, (
        f"REGRESSION: CI [{lo:.3f},{hi:.3f}] outside frozen {FROZEN_CI}")
    print("  PASS — backend reproduces the frozen GPT-2 O_h.\n")
    return mean


def test_llama_smoke(model_name, device="cpu"):
    print("=" * 70)
    print(f"[P0-B] Llama/Mistral smoke: {model_name}")
    print("=" * 70)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    be = get_backend(model)
    assert isinstance(be, LlamaBackend), f"expected LlamaBackend, got {type(be).__name__}"
    geo = model_head_geometry(model)
    print(f"  backend = LlamaBackend  geometry = {geo}")

    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)

    deep = geo["n_layer"] - 1
    for mode in ("query", "kv"):
        res = collect_residuals(model, ids, 256, device, (deep,), n_blocks=4,
                                max_points=800, group_mode=mode)
        keys = sorted(res)
        exp = geo["n_q"] if mode == "query" else geo["n_kv"]
        assert len(keys) == exp, f"{mode}: expected {exp} heads, got {len(keys)}"
        E0 = res[keys[0]]
        assert E0.shape[1] == geo["d_head"], f"d_head mismatch: {E0.shape[1]} vs {geo['d_head']}"
        dims = [twonn_dimension(res[k]) for k in keys]
        md = statistics.mean(dims)
        print(f"  group_mode={mode:>5}: {len(keys)} heads, ε shape [*,{E0.shape[1]}], "
              f"mean TwoNN={md:.2f}  (range {min(dims):.1f}-{max(dims):.1f})")
        assert 1.0 < md < 40.0, f"{mode}: implausible TwoNN mean {md:.2f}"
    print("  PASS — LlamaBackend runs end-to-end with sane residual geometry.\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Phase 0 regression gate")
    p.add_argument("--llama", type=str, default=None,
                   help="optional HF id of a small Llama/Mistral to smoke-test")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    test_gpt2_regression(device=args.device)
    if args.llama:
        test_llama_smoke(args.llama, device=args.device)
    print("Phase 0 gate complete.")
