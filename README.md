# ATLAS — *Attention's Latent Atlas of Specialized heads in Transformers*

> Formerly **NQP** (*Natural Quantization via State Preparation*). The founding quantization
> hypothesis was refuted; the project's destination turned out to be **the geometry it uncovered
> on the way**: an atlas of head-specific, mutually non-aligned manifolds in the residual of
> transformer attention — found across four autoregressive families (GPT-2, Qwen, Llama, Mistral).
> The repository keeps the name **NQP** on disk for git continuity, but the project's *goal* — and
> this README — point at the atlas.

**Principal investigator:** Juan Pablo Chancay · jpcpol@gmail.com
**Started:** 2026-06-24 · **Target venue:** NeurIPS / ICLR (workshop)
**License:** see [LICENSE.md](LICENSE.md) — CC BY-NC 4.0 (docs/theory) + AGPL-3.0 (src)

---

## The arc in one sentence

The project began by asking *"can we quantize an LLM better by rotating its weights into the natural
Fisher basis (analogous to measuring in the Hamiltonian's eigenbasis)?"* — and ended up answering
*"that idea did not work, but refuting it revealed a new, reproducible geometric structure in
attention: each head lives in its own low-dimensional (~7–11D) manifold, and these manifolds are
mutually non-aligned (overlap O_h ≪ 1) across four autoregressive transformer families — GPT-2,
Qwen2.5, Llama-3.1, Mistral. The existence is architecture-robust; the magnitude clusters by
attention design (GPT-2/Qwen ≈ 0.28, Llama/Mistral ≈ 0.20)."*

| Original hypothesis (refuted) | Surviving result (positive) |
|---|---|
| Quantization in the Fisher basis P̂ beats GPTQ+AWQ+QuIP | The activation Fisher is rank ~2 → collapses onto the baselines. **Refuted.** |
| A better *deployment* method exists | What exists is a **representation-level** object, not a compression trick |
| The attention tail is compressible | A scale-invariant atlas of head-specific manifolds (O_h ≈ 0.28), geometrically real but **not functionally compressible** by a simple autoencoder (Q06) |

The document that closes this arc: [docs/retrospective_vs_original_goal.md](docs/retrospective_vs_original_goal.md)
and §0 of the paper ([docs/paper_draft.md](docs/paper_draft.md)).

---

## Objectives (north star)

- **Scientific objective (largely achieved):** understand the geometric organization of attention
  and its functional role. → *There is an atlas of head-specific manifolds, non-aligned, robust
  across scale, corpus, and architecture (GPT-2 / Qwen / Llama / Mistral).*
- **Applied objective (open):** determine whether this organization can be exploited to build more
  efficient, adaptive, or interpretable Transformers — **without repeating NQP's mistake**: the
  existence of a geometric structure does not imply it is exploitable.

---

## Repository layout

```
NQP/
├── README.md          ← this index
├── LICENSE.md         ← dual license (CC BY-NC 4.0 + AGPL-3.0)
├── CLAUDE.md          ← project context for assistance
├── docs/              ← papers, questions, retrospective, figures
├── theory/            ← mathematical formalization (operator, quantum map, uncertainty)
├── experiments/       ← roadmap and experiment specifications
└── src/               ← implementation and measurement scripts
```

---

## 📄 `docs/` — Papers, figures, and narrative

| File | What it is |
|---|---|
| [docs/paper_draft.md](docs/paper_draft.md) | **Complete preprint.** Central result: an atlas of non-aligned head-specific manifolds across four autoregressive families (O_h ≪ 1; magnitude clusters by attention design). §0 ties results to the original objective; §3.1b is the cross-architecture result; §5 Related Work, refs [1]–[23]. **Start here.** |
| [docs/cross_architecture_plan.md](docs/cross_architecture_plan.md) | Cross-architecture research + roadmap: prior-work positioning, GQA obstacles, staged Phase 0–3 with gates, and the recorded Phase 0/1/2 + d_local-control results (Case B). |
| [docs/paper_skeleton.md](docs/paper_skeleton.md) | Earlier paper skeleton (thermodynamic structure + intrinsic geometry). |
| [docs/research_questions.md](docs/research_questions.md) | Open research questions (Q01–Q06 and derivatives). |
| [docs/retrospective_vs_original_goal.md](docs/retrospective_vs_original_goal.md) | What the results mean against the founding hypothesis, with the outcome stated plainly. |
| [docs/figure_data.json](docs/figure_data.json) · [docs/phase2_results.json](docs/phase2_results.json) · [docs/phase2_control.json](docs/phase2_control.json) | Single sources of truth for the figures and the cross-architecture O_h runs (+ d_local control). |
| [docs/figures/](docs/figures/) | The paper's 9 figures (6 main + 3 supplementary). Fig 1 = H×H overlap matrix (iconic); Fig 6 = O_h across architectures. |

---

## 🧮 `theory/` — Mathematical formalization

| File | What it is |
|---|---|
| [theory/operator_formalization.md](theory/operator_formalization.md) | Formalization of the preparation operator P̂ = U (diagonalizes Fisher). Basis of the original hypothesis (quantization branch). |
| [theory/quantum_transformer_map.md](theory/quantum_transformer_map.md) | Systematic quantum-mechanics ↔ Transformers map. Origin of the successor line; sorts the analogy into decorative / inert / exact (softmax = Boltzmann). |
| [theory/uncertainty_principle.md](theory/uncertainty_principle.md) | Weight/activation uncertainty principle (NQP-U branch). U1a ✅ (bases do not commute, angle ≈49°), U1b ❌ (no operational bound). |

---

## 🧪 `experiments/` — Roadmap and specification

| File | What it is |
|---|---|
| [experiments/README.md](experiments/README.md) | Quantization experiments (EXP-001…003) — the original branch, now closed. |
| [experiments/ROADMAP.md](experiments/ROADMAP.md) | A→B→C roadmap of Fisher quantization and the record of its refutation (gates A-G1…A-G4, L2-error vs PPL). Key reading for **why** the original idea did not pass. |

---

## 💻 `src/` — Implementation and measurement

### Original branch — Fisher quantization (documented negative results)

| File | Role |
|---|---|
| [src/fisher.py](src/fisher.py) | EXP-001: diagonal Fisher quantizer (P̂ = I). Flat and dead diagonal. |
| [src/fisher_block.py](src/fisher_block.py) | Path A: block-wise Fisher with real rotation (P̂ = U ≠ I). Gate A-G4. |
| [src/uncertainty.py](src/uncertainty.py) | EXP-U01: measure the commutator [P̂_W, P̂_A] (weight/activation bases). |
| [src/pareto.py](src/pareto.py) | EXP-U02: Pareto frontier ε_W / ε_A. |

### Surviving branch — attention geometry and thermodynamics

| File | Role |
|---|---|
| [src/residual.py](src/residual.py) | Exact decomposition `Attn = a·V_{i*} + (1−a)·ε`; residual collection + SVD; Top-1/full/low-rank patches. |
| [src/intrinsic.py](src/intrinsic.py) | **TwoNN** intrinsic-dimension estimator (validated on swiss-roll) + linear PCA(90%) rank. |
| [src/manifold.py](src/manifold.py) | `atlas_test`: connectivity, local homogeneity, interpolation, subspace overlap. |
| [src/thermo.py](src/thermo.py) | Boltzmann observables (F, ⟨E⟩, C, T_eff, S_vn) from the same Z the attention already computes. |
| [src/crystallize.py](src/crystallize.py) | Top-k (hard selection) baseline + perplexity; measures the damage the geometry must repair. |
| [src/scaling.py](src/scaling.py) | Intrinsic dimension and crystallization depth L_c across scale. |
| [src/atlas_scaling.py](src/atlas_scaling.py) | Inter-head overlap O_h across the GPT-2 family (scale-invariance). |
| [src/atlas_robustness.py](src/atlas_robustness.py) | **Hardening of the central claim:** bootstrap CI of O_h + sensitivity to d_local / N / depth. |
| [src/atlas_intercorpus.py](src/atlas_intercorpus.py) | Inter-corpus control (WikiText-103 vs C4): O_h is a property of the model, not the corpus. |
| [src/residual_backends.py](src/residual_backends.py) | **Architecture-agnostic residual extraction** (GPT-2 / Llama / Mistral / Qwen2 backends; handles RMSNorm, RoPE, GQA). Lets the same O_h protocol run on any of the four families. |
| [src/atlas_crossarch.py](src/atlas_crossarch.py) | **Cross-architecture O_h** with the GQA intra/inter-group pair split + per-model d_local + fixed-d_local control + depth×seed robustness. The Phase 1/2 driver. |
| [src/atlas_intramodel.py](src/atlas_intramodel.py) | **Intra-model confounder control:** within one model, correlate each head's intrinsic dimension against its overlap (Spearman + permutation p). Discriminates d_head vs d_int as the cross-arch driver — without retraining. |
| [src/autoencoder.py](src/autoencoder.py) | **EXP-Q06:** per-head nonlinear autoencoder (64→7→64) vs PCA rank-7. Clean negative: the manifold is not functionally compressible by this route. |
| [src/figure_data.py](src/figure_data.py) | Collects the figure data → `docs/figure_data.json`. |
| [src/make_figures.py](src/make_figures.py) | Renders the 9 figures → `docs/figures/` (incl. Fig 6, cross-architecture). |
| [tests/test_phase0_regression.py](tests/test_phase0_regression.py) | Regression gate: the backend refactor must reproduce GPT-2's O_h = 0.284 bit-for-bit. |

---

## Reproducing the figures

```bash
cd src
python figure_data.py     # → docs/figure_data.json (collects matrices fresh, ~minutes on CPU)
python make_figures.py     # → docs/figures/*.png
```

Models: the GPT-2 family (124M / 355M / 774M, via `transformers`). Data: WikiText-103 validation
(+ C4 for the inter-corpus control). All cross-scale comparisons fix N / number of heads / relative
depth.

---

## Status

| Component | Status |
|---|---|
| Fisher quantization (NQP-C1) | ❌ Refuted — collapses onto GPTQ+AWQ+QuIP |
| Uncertainty principle (NQP-U1) | ⚠️ Partial — bases do not commute but with no operational consequence |
| Scale-invariant atlas within GPT-2 (O_h ≈ 0.28) | ✅ Central result |
| Per-head low-dim nonlinear manifold (~7–11D) | ✅ |
| **Cross-architecture (GPT-2 / Qwen / Llama / Mistral)** | ✅ **Case B** — O_h ≪ 1 universal; magnitude clusters by attention design (d_local control confirms it is real geometry) |
| Functional compression of the manifold (Q06) | ❌ Honest negative (AE ≈ PCA) |
| Preprint | ✅ Complete draft (title promoted; cross-arch §3.1b + Fig 6 integrated) |
| Bibliographic metadata confirmation | ⬜ Pending (reference manager; refs [17]–[23] flagged) |
| Intra-model confounder control (d_head vs d_int) | ✅ Done — d_head is **confounded** with intrinsic dimension (Qwen ρ=−0.53, p=3e-4; GPT-2 same sign, n.s.). Lead demoted to "leading suspect" |
| Architectural ablation — *what component sets O_h?* | ⬜ Open (d_head is the lead suspect but confounded with d_int; ablation must track d_int as mediator; see Future Work) |

---

## Future directions (prioritized, with NQP's caution)

1. **What architectural component sets O_h?** — the cross-architecture result turned the existence
   question into this sharper one. The lead from our four points is **head dimension** (d_head 64 →
   ≈0.28, d_head 128 → ≈0.20, even though Qwen already has GQA/RoPE/RMSNorm) — but an intra-model
   control showed d_head is **confounded with intrinsic dimension**, which predicts overlap
   head-by-head in at least one family (Qwen ρ=−0.53). So the matched-scale ablation over {MHA↔GQA,
   #KV heads, d_head, RoPE↔learned, RMSNorm↔LayerNorm} must vary d_head while **tracking d_int as a
   mediator**, not treat d_head as the isolated cause.
   *Caveat (the NQP lesson): this establishes architecture→O_h, not O_h→quality.*
2. **Geometric routing across heads** — dynamic activation of a subset of heads (MoE-like, but by
   latent geometry rather than learned logits).
3. **Diagnostic metrics** — use O_h to detect head collapse/redundancy; requires no architectural
   change.
4. **Atlas stability under fine-tuning**; cross-*paradigm* generalization (encoder, encoder–decoder,
   state-space); **(speculative)** structured compression, geometric regularization.

The central medium-term question: **is the geometry causal or merely descriptive?**
