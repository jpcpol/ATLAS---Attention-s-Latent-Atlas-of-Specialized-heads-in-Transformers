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
  *bit-for-bit* (regression test). On a tiny Llama-style model it must run end-to-end and return sane
  TwoNN dims (single-digit per head, like GPT-2). *No refactor ships until GPT-2 numbers are
  unchanged.*

- **✅ Phase 0 DONE (2026-06-26).** Implemented `src/residual_backends.py` (`GPT2Backend` +
  `LlamaBackend`, dispatch via `get_backend`; RoPE/repeat_kv resolved per-family so it covers
  Llama / Mistral / Qwen2). `intrinsic.collect_residuals` is now a thin wrapper; all 8 consumers
  untouched. Gate `tests/test_phase0_regression.py`:
  - **P0-A (GPT-2 regression): PASS, exact** — O_h = 0.284, 95% CI [0.276, 0.292], bit-for-bit.
  - **P0-B (GQA smoke):** ran on **Qwen2.5-0.5B** (Llama-3.2-1B access still pending Meta approval).
    Qwen geometry n_q=14, n_kv=2, n_rep=7, d_head=64. Backend handled it correctly in both modes:
    `group_mode=query` → 14 heads, mean **TwoNN ≈ 6.3**; `group_mode=kv` → 2 groups, mean ≈ 5.8.
    First cross-architecture signal that the **low intrinsic dimension of the residual (~6–7D)
    survives RMSNorm + RoPE + GQA** — not a GPT-2 artifact. (O_h proper is measured under the full
    protocol in Phase 1/2; this gate only validates the extractor mechanics.)

### Phase 1 — Single small GQA model (the cheap falsifier) *(highest information/cost)*
- Run on **one** 1–3B GQA model end-to-end: per-head TwoNN → set d_local; O_h + bootstrap CI;
  intra-group vs inter-group split.
- **Gate G1 (the scientific decision):**
  - O_h ≪ 1 (CI upper bound well below, say, 0.6) → **continue to Phase 2** (Case A/B alive).
  - O_h ≈ 1 → **stop and investigate** (Case C): is it real, or a GQA/RoPE extraction bug? Re-derive
    by hand on 2 heads before believing it. This is the NQP discipline — a surprising positive is a
    bug until proven otherwise.

- **✅ Phase 1 DONE (2026-06-26) — GATE G1 PASS.** Ran `src/atlas_crossarch.py` on **Qwen2.5-0.5B**
  (deepest layer L23, N=1200, group_mode=query, d_local from the model's own TwoNN). The script
  introduces the GQA pair split (`kv_group_of`, `partition_pairs`; KV group = h // n_rep, verified
  against HF `repeat_kv` and unit-tested for pair counts 91/42/49).

  | population | O_h (d_local=7) | 95% bootstrap CI | pairs | reading |
  |---|---|---|---|---|
  | **inter-group** (independent V — the honest comparand vs GPT-2) | **0.290** | [0.284, 0.297] | 49 | ≈ GPT-2's 0.284 [0.276, 0.292] |
  | intra-group (shared V) | 0.482 | [0.453, 0.514] | 42 | inflated by shared value space, as predicted |
  | global (all pairs) | 0.379 | [0.355, 0.404] | 91 | contaminated by intra — would have *over-concluded* |
  | per-head TwoNN | ~7.1 | (range 4.1–26) | — | same ~7D as GPT-2 |

  **Anti-NQP discipline visible:** the naive *global* average (0.379) would have said "Qwen heads are
  more aligned than GPT-2"; the split shows that is a GQA artifact, and the honest inter-group number
  (0.290) matches GPT-2 (0.284). Not a bug: the extractor *does* distinguish geometry (intra 0.482 ≠
  inter 0.290), TwoNN≈7 is computed independently and also matches, and the d_local sweep replicates
  the GPT-2 shape (0.23→0.35, spread 0.114; conclusion O_h ≪ 1 is k-independent). **Preliminary Case
  A:** O_h ≈ 0.28–0.29 *and* TwoNN ≈ 7 survive RMSNorm + RoPE + aggressive GQA (n_rep=7). One model is
  a point, not a family — Phase 2 (Llama/Mistral) is still required before any title change.

### Phase 2 — Llama-3-8B + Mistral-7B under the matched protocol *(the headline run)*
- Both models, deepest-layer + last-3 band, N=1200, bootstrap CI, both O_h(32 Q) and O_h(8 KV),
  d_local from each model's TwoNN, d_local sweep for the caveat.
- Inter-corpus optional (WikiText already shown corpus-robust on GPT-2; one corpus is enough here).
- **Gate G2 (which claim we earn):**
  - all O_h in ~0.27–0.29, CIs overlapping GPT-2 → **Case A** → strong cross-family claim.
  - all ≪ 1 but spread → **Case B** → "existence robust, magnitude architecture-dependent."
  - divergent/≈1 anywhere → **Case C** → report honestly; the non-alignment becomes a GPT-2-family
    property and the paper's scope statement stands as-is.

- **✅ Phase 2 DONE (2026-06-26) — GATE G2 = CASE B.** Ran `src/atlas_crossarch.py` on a Colab T4
  (fp16; results in `docs/phase2_results.json`). A GPT-2 fp16/GPU regression first confirmed
  the precision/device do not move O_h (0.283 vs frozen 0.284), so the 7-8B numbers are trustworthy.
  Llama-8B needed `device_map="auto"` with a `max_memory` cap (full fp16, **no quantization** — CPU
  offload of overflow layers; the geometry is unaltered).

  | model | family | inter-group O_h | 95% CI | TwoNN | d_local |
  |---|---|---|---|---|---|
  | GPT-2 | MHA | 0.283 | [0.276, 0.290] | 7.1 | 7 |
  | Qwen2.5-0.5B | GQA 14/2 | 0.290 | [0.284, 0.297] | 7.1 | 7 |
  | Llama-3.1-8B | GQA 32/8 | 0.248 | [0.247, 0.250] | 10.8 | 11 |
  | Mistral-7B-v0.1 | GQA 32/8 | 0.226 | [0.224, 0.227] | 9.3 | 9 |

  **Verdict — Case B (the pre-registered "more interesting" outcome):**
  1. **Non-alignment is universal:** all four O_h ∈ [0.23, 0.29], all ≪ 1. The atlas of non-aligned
     per-head manifolds survives MHA *and* GQA, LayerNorm *and* RMSNorm, learned positions *and*
     RoPE. **Case C is firmly ruled out.**
  2. **Magnitude is architecture-dependent, and interpretably so:** Llama (0.248) and Mistral (0.226)
     — which share the *identical* attention geometry 32 Q / 8 KV / d_head 128 — fall **together and
     below** GPT-2/Qwen (~0.28), with non-overlapping CIs. Mistral is **not** an outlier; the
     Llama/Mistral attention shape produces more mutually-orthogonal heads. New question this opens:
     *why does GQA-32/8 (more heads, wider d_head) yield more orthogonal heads?*
  3. **Honest caveat:** TwoNN rises with the model (7→9→11), so d_local was re-derived per model
     (7/9/11). The residual of the larger models lives on a slightly higher-dimensional manifold —
     but still low (~11/128 ≈ 9% of ambient). The non-alignment conclusion is d_local-independent
     (the per-model d_local sweeps all stay ≪ 1), exactly as in the GPT-2 paper.

  **Claim earned (NOT "architecture-independent" — that would be Case A):** *the atlas of non-aligned
  per-head residual manifolds is a property of autoregressive transformers (GPT-2, Qwen, Llama,
  Mistral); the non-alignment O_h ≪ 1 is universal, while its exact magnitude depends on the
  attention architecture.* Distinguishing the invariant (existence) from the variable (magnitude) is
  scientifically stronger than a flat Case A.

- **✅ d_local CONTROL DONE (2026-06-26) — Case B survives, and strengthens.** The "official" O_h
  uses a per-model d_local (= each model's TwoNN: 7/7/9/11), so the cross-model comparison mixed
  points on different O_h(k) curves — a potential confound (O_h rises with d_local). We re-measured
  all four models at a **common fixed d_local = 7** (`atlas_crossarch` now reports `inter_O_h_k7`
  with bootstrap CI; results in `docs/phase2_control.json`):

  | model | O_h (per-model d_local) | O_h (fixed k=7) | 95% CI (k=7) |
  |---|---|---|---|
  | GPT-2 | 0.283 (k=7) | 0.283 | [0.276, 0.290] |
  | Qwen2.5-0.5B | 0.291 (k=7) | 0.291 | [0.285, 0.298] |
  | Mistral-7B | 0.226 (k=9) | 0.199 | [0.198, 0.201] |
  | Llama-3.1-8B | 0.248 (k=11) | 0.197 | [0.195, 0.199] |

  **Result:** at fixed k=7 the GPT-2/Qwen (~0.29) vs Llama/Mistral (~0.198) gap **widens** to
  ΔO_h ≈ 0.093 (CIs nowhere near overlapping). The variable d_local was *masking* part of the
  effect — Llama/Mistral used larger d_local (9/11), which inflated their O_h toward 0.226/0.248. So
  the architectural difference is **real geometry, not a d_local artifact** — this refutes the
  anti-NQP caveat we raised about the metric. Two consequences: (i) Case B is hardened; (ii) the
  d_int↔O_h correlation (higher intrinsic dim ⇒ lower overlap; raised in the ChatGPT cross-check) is
  legitimate as an *observation* (it persists at fixed k), though still **not causal**. Llama (0.197)
  and Mistral (0.199) are nearly identical at fixed k — same 32/8/4/128 attention, different
  training/data, same O_h ⇒ O_h is set by the *attention architecture*, not model size.

- **✅ ROBUSTNESS CONTROL DONE (2026-06-26) — Case B clustering is stable across depth and seed.**
  Before the architectural ablation we hardened §3.1b against the "single layer / single seed" risk:
  `ax.robustness()` re-measures inter-group O_h at fixed d_local = 7 over the **3 deepest layers × 2
  seeds**, per-cell bootstrap CI, with a verdict comparing the within-model wobble to the cross-arch
  gap (≈ 0.08). Qwen ran locally (CPU); Mistral and Llama on the Colab T4 (Llama with CPU offload,
  full fp16).

  | model | mean O_h (k=7) | depth-spread | seed-spread | verdict |
  |---|---|---|---|---|
  | Qwen2.5-0.5B | 0.283 | 0.017 | 0.005 | STABLE |
  | Mistral-7B | 0.198 | 0.006 | 0.002 | STABLE |
  | Llama-3.1-8B | 0.196 | 0.003 | 0.001 | STABLE |

  **Result:** the largest within-model wobble is Qwen's 0.017 — ≈ 5× below the 0.08 cluster gap.
  Seeds are interchangeable (≤ 0.005); per-cell CIs are ±0.002; Llama and Mistral agree to 0.002.
  The d_head clustering (64 → ~0.28, 128 → ~0.20) is **not an artifact of the reported layer or
  seed**. §3.1b and Appendix E updated. The cross-architecture arc (Phase 0→1→2 + d_local + robustness)
  is now closed; next gate is the **architectural ablation** (what component sets O_h? d_head is the
  lead) — *but consult ChatGPT first, per §8.*

- **✅ INTRA-MODEL CONTROL DONE (2026-06-26) — d_head is confounded with d_int.** ChatGPT's "favourite"
  cheap control (no retraining): *within* a single model, correlate each head's intrinsic dimension
  d_int_h against its mean inter-group overlap Ō_h (Spearman ρ, permutation p, pooled over 3 deep
  layers, d_local=7). `src/atlas_intramodel.py`; data `docs/intramodel_{gpt2,qwen}.json`.

  | model | d_head | n | ρ(d_int_h, Ō_h) | perm p | reading |
  |---|---|---|---|---|---|
  | GPT-2 | 64 | 36 | −0.26 | 0.13 | same sign, not significant |
  | Qwen2.5-0.5B | 64 | 42 | −0.53 | 0.0003 | significant negative |

  **Result:** in Qwen the d_int↔O_h link is clearly *per-head* (higher d_int ⇒ lower overlap, p=3e-4),
  not only between-architecture; GPT-2 shows the same sign without significance (n=36, low power). So
  **d_int is a real confounder for the d_head lead** — the cross-arch magnitude must NOT be attributed
  to d_head until the two are disentangled. Consequence for the ablation: it must vary d_head while
  **tracking d_int as a mediator**, not treat d_head as the isolated cause. §3.1b + §7 + new Appendix F
  updated to this cautious framing (lead demoted "implicating" → "leading suspect, confounded").

- **✅ O_h(k) SWEEP DONE (2026-06-26) — the cluster gap is a vertical offset of the whole curve.**
  Full k=4..10 sweep for GPT-2 and Qwen (the d_head-64 cluster); for Llama/Mistral only the k=7 +
  per-model-TwoNN points (fine sweep not run — would need Colab; slope+offset already clear, so the
  extra compute is low-yield, anti-NQP).

  | k | GPT-2 | Qwen | gap64 (cohesion) |
  |---|---|---|---|
  | 4 | 0.210 | 0.232 | 0.022 |
  | 7 | 0.284 | 0.290 | 0.006 |
  | 10 | 0.340 | 0.346 | 0.006 |

  **Result:** the two d_head-64 models trace a *common* O_h(k) curve (max cohesion gap 0.022, usually
  ≤0.006) → same régime, not just same point. The d_head-128 cluster is below it wherever compared:
  gap 0.089 at k=7, and crucially **Mistral@k=9 (0.226) < Qwen@k=9 (0.325)** and Llama@k=11 (0.248) <
  Qwen@k=10 (0.346) — the 128 curve is shifted DOWN at every k, not crossing. So the separation is not
  a k=7 artifact. Note added to §3.1b. Combined with the intra-model control: the gap is real geometry
  (sweep), but cannot yet be named a "d_head effect" because d_head and d_int are entangled (intra).

- **✅ CHEAP EXPERIMENT DONE (2026-06-26) — model scale is NOT the lever.** Since d_int is emergent
  (Valeriani 2302.00294: ID emerges through training, not a hyperparameter), the converse no-retraining
  intervention: hold d_head FIXED (=64, GPT-2 family) and vary size, at a fixed *relative* depth (0.9 —
  the ID profile's phases are located by relative, not absolute, depth). `src/atlas_dhead_control.py`,
  data `docs/dhead_control.json`.

  | model | layers | O_h | plateau d_int (rel 0.9) | peak d_int |
  |---|---|---|---|---|
  | gpt2 | 12 | 0.278 | 7.91 | 9.08 |
  | gpt2-medium | 24 | 0.280 | 8.00 | 9.66 |
  | gpt2-large | 36 | 0.279 | 7.85 | 10.53 |
  | spread | | 0.002 | 0.15 | 1.45 |

  **Result:** with d_head fixed, O_h (spread 0.002) and plateau d_int (0.15) are FLAT across 3× depth;
  only the PEAK d_int grows with size (1.45) — exactly Valeriani's "peak grows, plateau ~constant". So
  the intervention we wanted (move d_int with d_head fixed) is not realizable by scale: within a fixed
  d_head the plateau d_int doesn't move, so it can't move O_h. Consequences: (i) the
  d_head↔plateau-d_int↔O_h coupling is NOT a scale effect; (ii) O_h tracks the *plateau* d_int, not the
  peak — disambiguating *which* d_int matters. Rules scale out as the lever; sharpens the ablation to
  d_head. (Searched the literature first to vet the ChatGPT framing: post-treatment mediation confirmed
  spurious → not in paper; d_int-as-emergent confirmed [Valeriani]; the "latent Λ" intuition refined to
  the concrete peak/plateau split.)

### Reframed research question (post-controls, 2026-06-26)

The four controls (robustness, sweep, intra-model, cheap-experiment) + literature reframe the agenda
from *"which hyperparameter sets O_h?"* to **"what is the minimal geometric quantity that jointly
organizes the plateau intrinsic dimension and O_h, and which architectural decisions modulate it?"**
The evidence so far: O_h is coupled to the *plateau* (deep-régime) d_int; that coupling is set by
d_head (between clusters) and NOT by scale (within a fixed d_head); d_head and plateau-d_int remain
confounded. Next: a matched-scale ablation varying d_head with d_model/n_KV/depth/data fixed, *measuring
plateau d_int post-hoc as a mediator* (exploratory only — post-treatment mediator, NOT a causal claim).

**→ Design APPROVED (2026-06-26): see `docs/ablation_design.md`.** Batch-1 mini matched-scale (4
models ~20M, vary d_head ∈ {32,64,128,256} as the (d_head, n_heads) package, 2 seeds, WikiText-103,
Colab T4), vetted in two adversarial rounds. Frame softened to **fixed-point-like** (not "universality
classes"). Gate 0 (atlas maturation) gates validity: a flat O_h from an immature model is INVALID, not
a refutation. Claims the *total effect* of the intervention, not causation, not O_h→quality. Factorial
batch-2 (separating d_head from n_heads) only if P1 is strong. Next step: development (training
harness).

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

## 7. Architectural exploitation — evaluation and revised order (2026-06-26)

After the Qwen TwoNN≈6 signal we evaluated four candidate architectures for exploiting the atlas
(proposed in adversarial collaboration; J.P. Chancay is the final arbiter). The candidates:
**(A) Atlas-Routed Attention** (per-head encoder → router → top-k heads), **(B) shared base +
per-head charts** (M_i = Φ_i(M_base)), **(C) overlap regularization** (L_atlas = λ Σ O(h_i,h_j) or
(O−c)²), **(D) Dynamic Head MoE**.

**Decision and caveats (the points that keep us from repeating NQP):**

1. **"Heads are geometric experts" is a hypothesis, not a measured fact.** O_h≈0.28 says the
   subspaces are non-aligned; it does **not** say each head computes a distinct *function*. The
   decoupling could be optimization-induced (interference minimization) with no semantic content —
   the causal-vs-descriptive question (paper §4). Building A/D on the "expert" assumption risks a
   router with nothing to route. **Prerequisite before any routing architecture: measure whether the
   specialization is *functional*, not just geometric** (e.g. targeted ablation — does skipping the
   most mutually-aligned heads damage *less* than skipping the most orthogonal ones?).

2. **Architecture B is already in tension with our own data.** If a shared M_base existed and heads
   were charts of it, head-centering should have collapsed the pooled dimension — it did **not**
   (§3.1), and O_h≈0.28 (far from 1) says the subspaces are genuinely distinct, not offset-charts of
   a common base. B survives only if Φ_i is *nonlinear* (not a linear base/offset change) — i.e. the
   multi-month research, not a near-term bet.

3. **Architecture C's stated risk is the fatal one.** "We don't know if lower overlap implies better
   performance" — it might *worsen* it. If the atlas is a *consequence* of good training rather than a
   *cause*, regularizing O_h→c imposes the phenomenon's shadow, not its mechanism — exactly NQP-C1's
   error (forcing the Fisher basis while ignoring what Fisher optimizes). **C must not ship without
   at least a correlational signal that O_h moves with quality.**

**Revised roadmap (Phase A–D, exploitation track — only after cross-arch validation):**
- **Phase A (no architecture change):** measure the atlas *during training*, *during fine-tuning*,
  and *across Llama/Mistral*; **plus** the new functional-specialization test (1). Decides whether
  routing (A/D) has real substrate or only descriptive geometry (B/C) is honest.
- **Phase B:** overlap regularization — cheapest experiment, **gated on** a prior correlational
  signal from Phase A (caveat 3).
- **Phase C:** head router (ARA) — only if Phase A shows functional specialization.
- **Phase D:** full Atlas Transformer — only if A–C work.

**The bigger reframe (shared across collaborators):** if Llama confirms TwoNN≈6–7, the central
question shifts from *"how to exploit the atlas?"* to *"why does training Transformers inevitably
produce a ~6–7D atlas?"* — which would make it an **emergent representational law**, not a GPT-2
quirk. That is the higher-value scientific target, and it does not require an applied payoff to be
worth pursuing.

## 8. Collaboration protocol (for the record)

Research decisions in this project are cross-checked with a second model (ChatGPT) that brings a
distinct research perspective. **Governance, fixed:** J.P. Chancay is the sole final arbiter and
interlocutor; he sets the questions and makes the base/final decisions. Any future API-based
collaboration must be built as an **adversarial-collaboration harness**, not a consensus engine:
the second model receives the *same neutral factual context* (data, not our pre-formed conclusions),
produces an *independent* analysis, and both analyses are presented side-by-side with agreements and
disagreements highlighted for the human to decide. Rationale: the value is in *informed disagreement*
(which localizes uncertainty), not in easy convergence (an echo chamber). **Deferred until after the
cross-architecture validation** — it is meta-decision infrastructure, off the current critical path.
Security note: any OpenAI key lives in an environment variable, never on disk in the repo (cf. the
HF-token incident, now gitignored).

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
