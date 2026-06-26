"""
NQP-Q — Architecture-agnostic extraction of the attention residual ε  (Phase 0).

The frozen GPT-2 result is built on `intrinsic.collect_residuals`, which is hard-coupled
to GPT-2 internals (fused `c_attn`, `transformer.h[i].attn`, MHA with d_head=64). To take
the central claim O_h ≈ 0.28 cross-architecture (Llama / Mistral) we need the *same*
measurement on models that use separate q/k/v projections, RMSNorm, RoPE and GQA.

This module introduces a small backend abstraction with one invariant: the **output
contract is unchanged** — every backend returns

    dict[(layer:int, head:int)] -> Tensor[N, d_head]      (the residual cloud per head)

so every existing consumer (atlas_robustness, atlas_scaling, scaling, manifold,
autoencoder, atlas_intercorpus, figure_data) keeps working untouched. `collect_residuals`
in intrinsic.py becomes a thin wrapper over `get_backend(model).collect(...)`.

Residual definition (identical across backends):
    Attn_h = Σ_i a_i V_i,   i* = argmax_i a_i,   ε_h = Attn_h − V_{i*}
i.e. the attention output minus its single most-attended value vector, per head.

GQA note (Llama/Mistral): n_q query heads share n_kv key/value heads (n_rep = n_q/n_kv).
The residual is computed per *query* head (n_q of them) after `repeat_kv`. Because four
query heads in a group draw from the SAME value space, we expose `group_mode`:
    - "query": one cloud per query head            (n_q clouds; GPT-2-comparable count)
    - "kv":    query-head clouds pooled by KV group (n_kv clouds; independent value spaces)
Both are reported in Phase 2; the intra/inter-group split lives in the analysis scripts.
"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# helpers shared by backends
# ---------------------------------------------------------------------------

def _subsample(E: Tensor, max_points: int) -> Tensor:
    if E.shape[0] > max_points:
        idx = torch.randperm(E.shape[0])[:max_points]
        E = E[idx]
    return E


def _rotary_helpers(model):
    """Fetch (apply_rotary_pos_emb, repeat_kv) from the model's OWN modeling module.

    Llama / Mistral / Qwen2 share an identical attention layout but each defines these
    helpers in its own `modeling_<family>` module. We resolve them from the attention
    class's module so RoPE variant and any family-specific detail stay correct; we fall
    back to the Llama implementations (the canonical ones) if a family omits them.
    """
    import importlib
    from transformers.models.llama.modeling_llama import (
        apply_rotary_pos_emb as llama_rope, repeat_kv as llama_repeat)
    attn_cls = type(model.model.layers[0].self_attn)
    mod = importlib.import_module(attn_cls.__module__)
    rope = getattr(mod, "apply_rotary_pos_emb", llama_rope)
    repeat = getattr(mod, "repeat_kv", llama_repeat)
    return rope, repeat


def _residual_from_weights(p: Tensor, v: Tensor) -> Tensor:
    """
    Given attention weights p [H, T, T] and per-head values v [H, T, dh] (already
    repeated to H query heads), return ε [H, T, dh] = (p @ v) − v[argmax].
    """
    ctx = p @ v                                              # [H, T, dh]
    i_star = p.argmax(dim=-1)                                # [H, T]
    H, T, dh = v.shape
    v_star = torch.gather(v, 1, i_star.unsqueeze(-1).expand(H, T, dh))
    return ctx - v_star                                      # [H, T, dh]


# ---------------------------------------------------------------------------
# backend contract
# ---------------------------------------------------------------------------

class ResidualBackend(ABC):
    """Extracts per-(layer, head) attention-residual clouds from a model."""

    @abstractmethod
    def collect(self, model, ids, seq_len, device, layers, n_blocks=4,
                max_points=3000, group_mode="query") -> dict:
        ...


# ---------------------------------------------------------------------------
# GPT-2 backend — byte-for-byte the original collect_residuals logic
# ---------------------------------------------------------------------------

class GPT2Backend(ResidualBackend):
    """MHA, fused c_attn, learned positions, LayerNorm. `group_mode` is ignored
    (MHA already has one value space per head)."""

    def collect(self, model, ids, seq_len, device, layers, n_blocks=4,
                max_points=3000, group_mode="query") -> dict:
        n_head = model.config.n_head
        bufs: dict = {}
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
                eps = _residual_from_weights(p[0], v[0])         # [H,T,dh]
                for h in range(n_head):
                    bufs.setdefault((li, h), []).append(eps[h].detach())
            return hook

        for li in layers:
            handles.append(model.transformer.h[li].attn.register_forward_hook(make_hook(li)))
        _run_blocks(model, ids, seq_len, device, n_blocks)
        for hd in handles:
            hd.remove()
        return {k: _subsample(torch.cat(c, 0), max_points) for k, c in bufs.items()}


# ---------------------------------------------------------------------------
# Llama-style backend — separate q/k/v, RMSNorm, RoPE, GQA
# (Llama, Mistral, Qwen2 — identical attention layout; RoPE helpers resolved
#  per-family via _rotary_helpers)
# ---------------------------------------------------------------------------

class LlamaBackend(ResidualBackend):
    """
    Wraps each target `self_attn.forward` to capture ε, then delegates to the
    original forward so the model's own output is untouched. Handles any
    Llama-style attention (Llama / Mistral / Qwen2): the layout is identical and
    family-specific RoPE/repeat_kv are resolved from the model's own module.

    Mirrors transformers' `eager_attention_forward` exactly:
        q,k,v via q_proj/k_proj/v_proj  ->  RoPE on q,k (cos,sin passed in)
        repeat_kv(k,v, n_rep)  ->  scores = q·kᵀ * scaling + mask  ->  softmax
        ε = (p @ v) − v[argmax],  per query head.
    For group_mode="kv", the n_rep query heads sharing a KV group are concatenated
    into that group's cloud (n_kv clouds).
    """

    def collect(self, model, ids, seq_len, device, layers, n_blocks=4,
                max_points=3000, group_mode="query") -> dict:
        apply_rotary_pos_emb, repeat_kv = _rotary_helpers(model)

        assert group_mode in ("query", "kv")
        bufs: dict = {}
        originals = {}

        def make_fwd(li, attn):
            orig = attn.forward
            n_rep = attn.num_key_value_groups
            scaling = attn.scaling

            def fwd(hidden_states, position_embeddings=None, attention_mask=None,
                    past_key_values=None, **kw):
                B, T, _ = hidden_states.shape
                hs = (*hidden_states.shape[:-1], -1, attn.head_dim)
                q = attn.q_proj(hidden_states).view(hs).transpose(1, 2)   # [B,nq,T,dh]
                k = attn.k_proj(hidden_states).view(hs).transpose(1, 2)   # [B,nkv,T,dh]
                v = attn.v_proj(hidden_states).view(hs).transpose(1, 2)   # [B,nkv,T,dh]
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)
                kr = repeat_kv(k, n_rep)                                  # [B,nq,T,dh]
                vr = repeat_kv(v, n_rep)                                  # [B,nq,T,dh]
                scores = (q @ kr.transpose(-1, -2)) * scaling            # [B,nq,T,T]
                cmask = torch.tril(torch.ones(T, T, device=hidden_states.device,
                                              dtype=torch.bool))
                scores = scores.masked_fill(~cmask, float("-inf"))
                p = scores.softmax(dim=-1)
                eps = _residual_from_weights(p[0], vr[0])                 # [nq,T,dh]
                nq = eps.shape[0]
                if group_mode == "query":
                    for h in range(nq):
                        bufs.setdefault((li, h), []).append(eps[h].detach())
                else:  # "kv": pool the n_rep query heads of each KV group
                    n_kv = nq // n_rep
                    for g in range(n_kv):
                        grp = eps[g * n_rep:(g + 1) * n_rep]             # [n_rep,T,dh]
                        bufs.setdefault((li, g), []).append(
                            grp.reshape(-1, grp.shape[-1]).detach())     # [n_rep*T,dh]
                return orig(hidden_states, position_embeddings=position_embeddings,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values, **kw)
            return fwd

        for li in layers:
            attn = model.model.layers[li].self_attn
            originals[li] = attn.forward
            attn.forward = make_fwd(li, attn)
        try:
            _run_blocks(model, ids, seq_len, device, n_blocks)
        finally:
            for li, f in originals.items():
                model.model.layers[li].self_attn.forward = f
        return {k: _subsample(torch.cat(c, 0), max_points) for k, c in bufs.items()}


# ---------------------------------------------------------------------------
# shared forward driver + dispatch
# ---------------------------------------------------------------------------

def _run_blocks(model, ids, seq_len, device, n_blocks):
    nb = min(n_blocks, ids.numel() // seq_len)
    with torch.no_grad():
        for b in range(nb):
            blk = ids[b * seq_len:(b + 1) * seq_len].unsqueeze(0).to(device)
            model(input_ids=blk)


def get_backend(model) -> ResidualBackend:
    """Pick a backend from the model's structure. GPT-2 exposes `.transformer.h`;
    Llama/Mistral expose `.model.layers` with `self_attn.q_proj`."""
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return GPT2Backend()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        inner = model.model.layers[0].self_attn
        if hasattr(inner, "q_proj") and hasattr(inner, "k_proj"):
            return LlamaBackend()
    raise ValueError(
        f"No residual backend for {type(model).__name__}. "
        "Supported: GPT-2 (transformer.h) and Llama/Mistral (model.layers + self_attn.q_proj)."
    )


def model_head_geometry(model) -> dict:
    """Report head counts so callers can choose n_heads / group_mode sensibly."""
    cfg = model.config
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return {"family": "gpt2", "n_layer": cfg.n_layer, "n_q": cfg.n_head,
                "n_kv": cfg.n_head, "d_head": cfg.n_embd // cfg.n_head, "n_rep": 1}
    n_q = getattr(cfg, "num_attention_heads")
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    d_head = getattr(cfg, "head_dim", cfg.hidden_size // n_q)
    return {"family": "llama", "n_layer": cfg.num_hidden_layers, "n_q": n_q,
            "n_kv": n_kv, "d_head": d_head, "n_rep": n_q // n_kv}
