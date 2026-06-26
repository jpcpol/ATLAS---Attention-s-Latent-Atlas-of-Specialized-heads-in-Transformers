# Cross-Architecture Extension — Research, Literature Comparison, and Roadmap

**Date:** 2026-06-26 · **Researcher:** Juan Pablo Chancay
**Goal:** promote the central claim from *"scale-invariant within the GPT-2 family"* to a stronger,
architecture-aware statement by measuring the inter-head residual overlap O_h on **Llama** and
**Mistral** under a matched protocol.

This document is **research + planning only** — no measurement code is changed here. It records (1)
what the literature already establishes, (2) where our result is genuinely novel, (3) the
architectural obstacles, and (4) a staged roadmap with explicit go/no-go gates and the NQP caution
(*a geometric structure existing does not mean it is exploitable, nor that it is novel — verify
both*).

---

## 1. What we are extending

Our frozen result (see [paper_draft.md](paper_draft.md) §3.1): on the residual ε = Attn − a·V_{i\*},
the mean inter-head subspace overlap is **O_h ≈ 0.28** (95% bootstrap CI [0.27, 0.29]),
statistically indistinguishable across GPT-2 small/medium/large (124M→774M) and across corpora
(WikiText vs C4). The metric is the **mean of the principal-angle cosines** between per-head
d_local-dimensional SVD frames of the residual cloud, measured **between heads of the same layer**.

The open question (from `tarea.txt`): is **0.28 a GPT-2-family constant, or an autoregressive-
transformer constant?** Three a priori outcomes:

- **Case A** — Llama ≈ Mistral ≈ 0.27–0.29 → *"architecture-independent evidence for head-wise
  manifold organization … observed across autoregressive transformer families."*
- **Case B** — magnitude differs (e.g. 0.23 / 0.31) but all ≪ 1 → *"the existence of non-alignment
  is robust; its magnitude is architecture-dependent."* (Scientifically arguably **more**
  interesting.)
- **Case C** — Llama ≈ 0.95 (heads aligned) → would falsify universality. Considered unlikely given
  the convergent evidence below, but it is the result that would matter most, so the protocol is
  built to detect it honestly.

---

## 2. Literature comparison (what is already known vs. our novelty)

### 2.1 The closest prior work — and why it does not pre-empt us

**Projection Kernel for head affinity** (arXiv:2601.10266). This is the nearest neighbor to our
metric and the most important paper to position against. It measures subspace affinity between heads
via **principal angles** — the same mathematical machinery as O_h. Crucially, it differs on **four**
axes:

| Axis | Projection Kernel (2601.10266) | **Our O_h** |
|---|---|---|
| Object measured | attention **weight** matrices (W_Q,W_K,W_V,W_O) | attention **activations** — the residual ε |
| Relationship studied | **between layers** (h→h′ wiring, ℓ_h < ℓ_h′) | **between heads of the same layer** |
| Metric | Σ cos²(θ_i) (unnormalized sum) | mean cos(θ_i) (normalized) |
| Residual decomposition | none (full weights) | yes (Attn − dominant value) |
| Models | **GPT-2 small only** | GPT-2 family → (this plan) Llama/Mistral |
| Cross-arch / scale | none | the contribution |

→ **Our novelty is intact and sharper after this read:** an *activation-space, within-layer,
residual-specific, normalized, cross-architecture* measurement. We must cite 2601.10266 prominently
and state the distinction explicitly (it strengthens, not threatens, the framing).

### 2.2 Intrinsic dimension is already known to be cross-architecture consistent

- **Valeriani et al., "The geometry of hidden representations of large transformer models"**
  (arXiv:2302.00294, Laio group): whole-layer hidden representations follow an **expand-then-contract
  ("hourglone"/two-peak)** intrinsic-dimension profile across many models and modalities (protein,
  vision, language), via **TwoNN** — the same estimator we use. They study *whole-layer*
  representations, **not per-head, not the residual, not inter-head alignment.**
- Survey-level evidence: a peak TwoNN intrinsic dimension of **~19–22 is reported as consistent
  across ~6 models and architectures** (GPT/Llama variants) — the "universal hourglass."

→ **Implication for us:** the *whole-layer* ID being architecture-consistent is established prior
art. We must **not** sell "intrinsic dimension is scale-invariant" as our novelty — it is partly
known. Our distinct object is the **per-head residual** (~7D, far below whole-layer ID) **and the
inter-head non-alignment O_h**, neither of which these papers measure. This also predicts Case A/B
over Case C: if whole-layer geometry is already cross-architecture stable, per-head decoupling
plausibly is too.

### 2.3 Head redundancy / similarity in Llama — adjacent, not the same

- **CHAI — Clustered Head Attention** (arXiv:2403.08058): clusters heads with **similar output**
  in Llama for inference pruning. Operates on the **full head output via cosine**, not the residual
  ε and not principal angles between subspaces.
- **"What Matters in Transformers? Not All Attention Is Needed"** (arXiv:2406.15786) and **ShortGPT**
  (arXiv:2403.03853): later-layer attention sub-layers are the most redundant across model families;
  block-level cosine input/output similarity.
- Quantitative anchor: in **Llama-2-7B**, token-wise attention-module cosine similarity sits around
  **0.1–0.4** in early-mid layers — i.e. heads are *not* trivially aligned, consistent with our
  "decoupled modules" reading and with an expected O_h ≪ 1.

→ These works establish that **functional redundancy exists and is exploited**, but via output-level
cosine, between layers/blocks, for pruning. None measures the **within-layer residual subspace
geometry**. They are *related work / motivation*, not competitors to the claim.

### 2.4 Net positioning

> Prior work establishes (a) principal-angle head affinity on GPT-2 **weights** across layers,
> (b) architecture-consistent **whole-layer** intrinsic dimension, and (c) output-cosine head
> redundancy in Llama for pruning. **No prior work measures the within-layer, activation-space,
> residual-specific inter-head subspace non-alignment, nor its invariance across architectures.**
> That intersection is our contribution; the cross-architecture run is what converts it from a
> GPT-2 observation into a candidate transformer regularity.

---

## 3. Architectural obstacles (why this is not a one-line model swap)

Our extractor [src/intrinsic.py](../src/intrinsic.py) `collect_residuals` is **hard-coupled to
GPT-2**: it hooks `module.c_attn` (the **fused** QKV projection), reads `model.transformer.h[li].attn`,
and assumes `n_head` value-heads with `d_head = 64`. Llama/Mistral break every one of these
assumptions:

| Feature | GPT-2 | Llama-3 / Mistral | Consequence for O_h |
|---|---|---|---|
| QKV projection | fused `c_attn` | separate `q_proj`/`k_proj`/`v_proj`/`o_proj` | new hook target |
| Normalization | LayerNorm | RMSNorm (pre-norm) | hook input differs; residual stream cleaner |
| Positional | learned abs. | **RoPE** (applied to q,k) | must apply rotary before scores or ε is wrong |
| Heads | MHA, 12×64 | **GQA**: 32 Q-heads share **8 KV-heads** (G=4), d_head=**128** | **design decision below** |
| Module path | `transformer.h[i].attn` | `model.layers[i].self_attn` | new module path |

**The GQA decision (the one real conceptual choice).** With GQA, `ctx = p @ v` yields **32 query
outputs** but only **8 distinct value subspaces** (each KV-head shared by 4 Q-heads). The residual
ε = ctx − v_{i\*} is computed per **Q-head** (32 of them), but four Q-heads in a group draw from the
*same* V. Two valid measurements, to be reported transparently:

- **O_h over the 32 Q-heads** — directly comparable in *count-of-heads* spirit to GPT-2, but four
  heads per group share a value space, which could **deflate** O_h (artificially more alignment
  within a group). Must check intra-group vs inter-group overlap separately.
- **O_h over the 8 KV-heads** — the geometrically honest "number of independent value subspaces."
  Likely the cleaner cross-arch comparand, but fewer heads → wider bootstrap CIs.

→ **Plan: report both, and the intra-group vs inter-group split**, so a reviewer sees GQA is handled,
not hidden. d_head also changes 64→128, so `d_local = 7` (matched to intrinsic dimension) must be
**re-derived from TwoNN on each model**, not assumed (this is already our §2.3 discipline).

---

## 4. Methodological protocol (matched, from `tarea.txt`)

Do **not** re-run the full pipeline. Measure only what promotes the claim:

- **Mandatory:** O_h + bootstrap CI; per-head intrinsic dimension (TwoNN, to set d_local per model).
- **Optional:** the H×H overlap matrix (the iconic figure, if heads count allows).
- **Do NOT repeat:** Q02 (Top-k), Q03 (low-rank), Q06 (autoencoder) — those were about
  compressibility, already settled on GPT-2 and out of scope for the universality claim.

**Matched-protocol controls (report explicitly as a "matched relative-depth protocol"):**
- same **relative depth** (deepest layer, or last-3 band, by fraction — not absolute index, since
  Llama has 32 layers vs GPT-2's 12/24/36);
- same **N** residual points per head (e.g. 1200);
- same **estimation protocol** (centered SVD frame, mean principal-angle cosine, 2000-resample
  percentile bootstrap over head pairs);
- per-model **d_local = round(TwoNN intrinsic dim)**, with the d_local sweep reported (our §3.1
  caveat: absolute value depends on d_local; non-alignment O_h ≪ 1 does not).

**Compute reality.** GPT-2 ran on CPU. Llama-3-8B / Mistral-7B forward passes for ~1200 residuals ×
8 KV-heads × deepest layers are feasible on a single GPU (or CPU with `float16`/few blocks, slowly).
We are **inference/forward-only** (no training), so no optimizer memory — the binding constraint is
loading ~7–8B params (≈16 GB fp16). 7B is the right size; **do not jump to 70B** (same anti-pattern
as NQP's "jump to B/K-FAC"). If GPU is unavailable, a 1–3B GQA model (e.g. Llama-3.2-1B/3B) is a
valid first data point that still exercises RMSNorm+RoPE+GQA.

---

## 5. Development roadmap (staged, with go/no-go gates)

### Phase 0 — Refactor the extractor to be architecture-agnostic *(prerequisite, low risk)*
- Generalize `collect_residuals` into a small **backend abstraction**: given a model, return per-
  (layer, head) residual clouds. GPT-2 backend (existing) + a **Llama/Mistral backend** that hooks
  `self_attn`, reads `q_proj/k_proj/v_proj`, applies **RoPE** and the causal mask, does **`repeat_kv`**
  to expand 8→32, and computes ε per Q-head (and a KV-head aggregation).
- **Validation gate P0:** on GPT-2 the refactor must reproduce the frozen O_h = 0.284 [0.276, 0.292]
  *bit-for-bit* (regression test). On a tiny Llama (1B) it must run end-to-end and return sane
  TwoNN dims (single-digit per head, like GPT-2). *No refactor ships until GPT-2 numbers are
  unchanged.*

### Phase 1 — Single small GQA model (the cheap falsifier) *(highest information/cost)*
- Run on **one** 1–3B GQA model end-to-end: per-head TwoNN → set d_local; O_h + bootstrap CI;
  intra-group vs inter-group split.
- **Gate G1 (the scientific decision):**
  - O_h ≪ 1 (CI upper bound well below, say, 0.6) → **continue to Phase 2** (Case A/B alive).
  - O_h ≈ 1 → **stop and investigate** (Case C): is it real, or a GQA/RoPE extraction bug? Re-derive
    by hand on 2 heads before believing it. This is the NQP discipline — a surprising positive is a
    bug until proven otherwise.

### Phase 2 — Llama-3-8B + Mistral-7B under the matched protocol *(the headline run)*
- Both models, deepest-layer + last-3 band, N=1200, bootstrap CI, both O_h(32 Q) and O_h(8 KV),
  d_local from each model's TwoNN, d_local sweep for the caveat.
- Inter-corpus optional (WikiText already shown corpus-robust on GPT-2; one corpus is enough here).
- **Gate G2 (which claim we earn):**
  - all O_h in ~0.27–0.29, CIs overlapping GPT-2 → **Case A** → strong cross-family claim.
  - all ≪ 1 but spread → **Case B** → "existence robust, magnitude architecture-dependent."
  - divergent/≈1 anywhere → **Case C** → report honestly; the non-alignment becomes a GPT-2-family
    property and the paper's scope statement stands as-is.

### Phase 3 — Write-up integration *(only after G2)*
- Add a cross-architecture subsection + one figure (O_h vs architecture with CIs, analogous to Fig 2).
- **Title/abstract change only if Case A or B holds** (per `tarea.txt`: do not touch the title until
  the data is in). Candidate promotion: *"Evidence for a Scale-Invariant Atlas of Head-Specific
  Manifolds in Autoregressive Transformers."*
- Update Related Work to cite 2601.10266 (Projection Kernel), 2302.00294 (Valeriani et al.),
  2403.08058 (CHAI), 2406.15786 — with the explicit "we differ by …" sentences from §2.

---

## 6. Final-objective view (the north star this serves)

This run is **external-validity** work, not exploitation. It answers *"is the phenomenon robust
across architectures?"* — Level-1 of the long-term program (`tarea.txt`): *understand the geometric
organization of attention*. It deliberately does **not** attempt the applied Level-3 bets (geometric
routing, diagnostics, fine-tuning stability), which depend on this answer. The causal-vs-descriptive
question remains the medium-term open problem.

**Honest pre-commitment (anti-NQP):** a clean Case **B** or even a partial Case **C** is publishable
and must be reported as found. We are not running this to confirm 0.28; we are running it to *find
out whether 0.28 travels.* The protocol is pre-registered above so the result is not retrofitted to
the hypothesis.

---

## Sources (web research, 2026-06-26)

- [Measuring Affinity between Attention-Head Weight Subspaces via the Projection Kernel (arXiv:2601.10266)](https://arxiv.org/abs/2601.10266)
- [The geometry of hidden representations of large transformer models (arXiv:2302.00294)](https://arxiv.org/abs/2302.00294)
- [CHAI: Clustered Head Attention for Efficient LLM Inference (arXiv:2403.08058)](https://arxiv.org/abs/2403.08058)
- [What Matters in Transformers? Not All Attention Is Needed (arXiv:2406.15786)](https://arxiv.org/abs/2406.15786)
- [ShortGPT: Layers in LLMs are More Redundant Than You Expect (arXiv:2403.03853)](https://arxiv.org/abs/2403.03853)
- [Linear Predictability of Attention Heads in LLMs (arXiv:2603.13314)](https://arxiv.org/abs/2603.13314)
- [Scale-invariant Attention (OpenReview ONE9LYBQYS)](https://openreview.net/pdf?id=ONE9LYBQYS) — *sequence-length invariance, not model-scale; name-only overlap, no conflict (to re-confirm)*
- [HuggingFace Llama modeling reference (modeling_llama.py)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [GQA: Training Generalized Multi-Query Transformer Models (arXiv:2305.13245)](https://arxiv.org/abs/2305.13245)
