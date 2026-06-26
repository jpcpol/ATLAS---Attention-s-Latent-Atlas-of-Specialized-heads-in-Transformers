"""
NQP-Q — Cross-architecture inter-head overlap O_h  (Phase 1: the cheap falsifier).

Takes the frozen GPT-2 result (O_h ≈ 0.28, residual subspaces non-aligned) to a GQA
model and asks: does the non-alignment survive RMSNorm + RoPE + GQA? This is the
external-validity step from docs/cross_architecture_plan.md §5 (Phase 1), gated on G1.

WHY THIS IS NOT just atlas_robustness with a model swap — GQA changes the meaning of a
"head pair". With n_rep query heads sharing one KV head, three pair populations exist:

    intra-group  (g_i == g_j): the two query heads share the SAME value space; their
                  residual ε differs only through the attention weights p. Overlap is
                  expected to be INFLATED by the shared V — a measurement artifact.
    inter-group  (g_i != g_j): independent value spaces. This is the honest comparand
                  vs GPT-2, where MHA gives every head its own V (so *every* GPT-2 pair
                  is inter-group by construction). THIS is the number that goes in the
                  cross-architecture table.
    global       (all pairs): the naive average — contaminated by the intra-group pairs.

Reporting all three makes the anti-NQP discipline visible: the naive global average
over-concludes; the inter-group control reframes. Gate G1 is decided on inter-group.

KV-group of query head h:  g = h // n_rep.
  Verified against HF repeat_kv (= torch.repeat_interleave, dim=1): the expand+reshape
  lays heads out in contiguous blocks [kv0×n_rep, kv1×n_rep, ...], so the first n_rep
  query heads share KV head 0, etc. The LlamaBackend emits query heads in this same
  natural order, so key (layer, h) maps to KV group h // n_rep. (Asserted at runtime.)

This module also serves as the first reusable brick of the eventual architecture work:
`kv_group_of`, `partition_pairs`, and per-head overlap are exactly what a geometric
router / overlap regularizer will need.

Forward-only. Reuses pairwise_overlaps / bootstrap_ci / head_bases from atlas_robustness.
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

from intrinsic import collect_residuals, twonn_dimension
from atlas_robustness import head_bases, bootstrap_ci
from residual_backends import model_head_geometry


# ---------------------------------------------------------------------------
# GQA-aware pair partitioning (the one genuinely new piece of logic)
# ---------------------------------------------------------------------------

def kv_group_of(head_index: int, n_rep: int) -> int:
    """KV group of a query head, per HF repeat_kv (contiguous blocks)."""
    return head_index // n_rep


def partition_pairs(res_by_head: dict, d_local: int, n_rep: int):
    """
    Compute O(h_i, h_j) for every unordered query-head pair and split the pairs into
    intra-group / inter-group by their KV group. Returns dict of lists:
        {"global": [...], "intra": [...], "inter": [...]}.
    Keys of res_by_head are (layer, head); pairing is within a single layer.
    """
    bases = head_bases(res_by_head, d_local)
    keys = sorted(bases)                                   # [(layer, head), ...]
    out = {"global": [], "intra": [], "inter": []}
    for i, ki in enumerate(keys):
        for kj in keys[i + 1:]:
            s = torch.linalg.svdvals(bases[ki].t() @ bases[kj]).clamp(0, 1)
            o = s.mean().item()
            out["global"].append(o)
            same = kv_group_of(ki[1], n_rep) == kv_group_of(kj[1], n_rep)
            out["intra" if same else "inter"].append(o)
    return out


def _ci_line(name, vals):
    if not vals:
        return f"    {name:>7}: (no pairs)"
    mean, lo, hi, sd = bootstrap_ci(vals)
    return f"    {name:>7}: O_h = {mean:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  ({len(vals)} pairs)"


# ---------------------------------------------------------------------------
# loaders / collection
# ---------------------------------------------------------------------------

def _load_any(model_name, device, seed, dtype=None, offload=False):
    """Generic causal-LM loader (GPT-2 / Llama / Mistral / Qwen2) + WikiText-103.

    dtype defaults to float32 on CPU and float16 on cuda. A 7B fits in 16 GB VRAM in
    fp16; an 8B (Llama) does not on a 15 GB T4 — set offload=True to use
    device_map="auto", which keeps the model in full fp16 (NO quantization — a clean
    measurement, consistent with this project's anti-quantization origin) and spills
    the overflow layers to CPU RAM. Slower, but the geometry is unaltered. Subspace
    geometry is precision-robust; the GPU regression gate (GPT-2 fp16 → O_h≈0.284)
    verifies this rather than assuming it.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import os
    torch.manual_seed(seed)
    is_cuda = str(device).startswith("cuda")
    if not is_cuda:
        torch.set_num_threads(min(8, os.cpu_count() or 4))
    if dtype is None:
        dtype = torch.float16 if is_cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if is_cuda and offload:
        # auto-shard across GPU + CPU (full fp16). device_map="auto" alone will cram the
        # GPU until it OOMs on an 8B that *almost* fits; we cap GPU memory explicitly so
        # accelerate spills the overflow layers to CPU, leaving VRAM headroom for the
        # forward-pass activations. Inputs must enter on the embedding's device.
        import os
        gpu_gib = int(os.environ.get("NQP_GPU_GIB", "11"))   # leave ~3–4 GiB headroom
        cpu_gib = int(os.environ.get("NQP_CPU_GIB", "20"))
        max_mem = {0: f"{gpu_gib}GiB", "cpu": f"{cpu_gib}GiB"}
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map="auto", max_memory=max_mem,
            low_cpu_mem_usage=True)
        try:
            in_dev = next(model.parameters()).device
        except StopIteration:
            in_dev = torch.device(device)
        model._nqp_input_device = in_dev
    elif is_cuda:
        # whole model into VRAM; stream shards so system RAM is not the bottleneck.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, device_map=device, low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model.to(device)
    model.eval()
    ds = _load_wikitext103(load_dataset)
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)
    return model, ids


def _load_wikitext103(load_dataset):
    """WikiText-103 validation, robust to the datasets-library id change.

    Newer `datasets` requires the canonical `namespace/name` id and rejects the bare
    `wikitext` shorthand (HfUriError). Try the canonical id first, fall back to the
    legacy short name for older library versions.
    """
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            return load_dataset(repo, "wikitext-103-raw-v1")
        except Exception:
            continue
    # last resort: surface the canonical-id error
    return load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")


def _deep_query_heads(model, ids, layer, device, n_blocks, n_points):
    """All query-head residual clouds of one layer, capped at n_points each."""
    res = collect_residuals(model, ids, 256, device, (layer,),
                            n_blocks=n_blocks, max_points=n_points + 200,
                            group_mode="query")
    keys = sorted(k for k in res if k[0] == layer)
    avail = min(res[k].shape[0] for k in keys)
    N = min(avail, n_points)
    return {k: res[k][:N] for k in keys}, N


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(model_name="Qwen/Qwen2.5-0.5B", device="cpu", n_blocks=12,
        n_points=1200, seed=42, g1_threshold=0.6, offload=False):
    print(f"[Atlas-crossarch] {model_name}" + ("  [offload]" if offload else ""))
    model, ids = _load_any(model_name, device, seed, offload=offload)
    geo = model_head_geometry(model)
    n_rep = geo["n_rep"]
    deep = geo["n_layer"] - 1
    print(f"  geometry: family={geo['family']} n_layer={geo['n_layer']} "
          f"n_q={geo['n_q']} n_kv={geo['n_kv']} n_rep={n_rep} d_head={geo['d_head']}")

    # ---- collect residuals (deepest layer, all query heads) -----------------
    by_head, N = _deep_query_heads(model, ids, deep, device, n_blocks, n_points)
    assert len(by_head) == geo["n_q"], (len(by_head), geo["n_q"])
    print(f"  layer {deep}: {len(by_head)} query heads, N={N}")

    # sanity: verify the KV-group partition is non-degenerate
    n_kv = geo["n_kv"]
    if n_rep == 1:
        print("  NOTE: MHA model (n_rep=1) — every pair is inter-group (== global).")

    # ---- d_local from this model's intrinsic dimension ----------------------
    dims = [twonn_dimension(by_head[k]) for k in sorted(by_head)]
    mean_dim = statistics.mean(dims)
    d_local = max(2, round(mean_dim))
    print(f"  per-head TwoNN: mean {mean_dim:.2f} (range {min(dims):.1f}-{max(dims):.1f}) "
          f"=> d_local={d_local}")

    # ---- three O_h with bootstrap CI ----------------------------------------
    parts = partition_pairs(by_head, d_local, n_rep)
    print(f"\n  O_h at d_local={d_local} (bootstrap 95% CI):")
    print(_ci_line("global", parts["global"]))
    print(_ci_line("intra", parts["intra"]))
    print(_ci_line("inter", parts["inter"]))

    # the official cross-arch comparand
    inter = parts["inter"] if parts["inter"] else parts["global"]
    mean_i, lo_i, hi_i, _ = bootstrap_ci(inter)

    # ---- d_local sweep on the inter-group O_h (the paper's caveat) ----------
    print(f"\n  d_local sweep (inter-group O_h):")
    print(f"      {'k':>3} | {'O_h':>6}")
    sweep = {}
    for k in range(4, 11):
        p = partition_pairs(by_head, k, n_rep)
        seq = p["inter"] if p["inter"] else p["global"]
        sweep[k] = statistics.mean(seq)
        print(f"      {k:>3} | {sweep[k]:>6.3f}")
    spread_k = max(sweep.values()) - min(sweep.values())

    # ---- Gate G1 verdict (on inter-group) -----------------------------------
    print(f"\n{'='*70}\n[GATE G1] {model_name}\n{'='*70}")
    print(f"  inter-group O_h = {mean_i:.3f}  95% CI [{lo_i:.3f}, {hi_i:.3f}]")
    print(f"  intrinsic dim ~{mean_dim:.1f}  |  d_local sweep spread {spread_k:.3f}")
    if hi_i < g1_threshold:
        print(f"  PASS — inter-group O_h CI upper bound {hi_i:.3f} < {g1_threshold}: "
              f"non-alignment SURVIVES this architecture. Proceed to Phase 2.")
    elif lo_i > 0.85:
        print(f"  FAIL/INVESTIGATE — O_h ≈ 1 (Case C). Heads look aligned. "
              f"Treat as a GQA/RoPE extraction bug until verified by hand on 2 heads.")
    else:
        print(f"  AMBIGUOUS — O_h in the middle band. Inspect intra vs inter and the "
              f"d_local sweep before deciding.")
    return {"model": model_name, "geometry": geo, "N": N, "d_local": d_local,
            "mean_dim": mean_dim, "parts_summary": {
                k: (bootstrap_ci(v)[:3] if v else None) for k, v in parts.items()},
            "inter_O_h": mean_i, "inter_ci": (lo_i, hi_i), "dlocal_sweep": sweep,
            "dlocal_spread": spread_k}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="NQP-Q Phase 1 — cross-architecture O_h")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--n-blocks", type=int, default=12)
    p.add_argument("--n-points", type=int, default=1200)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    run(model_name=args.model, device=args.device,
        n_blocks=args.n_blocks, n_points=args.n_points)
