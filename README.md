# ATLAS — *Attention's Latent Atlas of Specialized heads in Transformers*

> Formerly **NQP** (*Natural Quantization via State Preparation*). The founding quantization
> hypothesis was refuted; the project's destination turned out to be **the geometry it uncovered
> on the way**: a scale-invariant atlas of head-specific, mutually non-aligned manifolds in the
> residual of transformer attention. The repository keeps the name **NQP** on disk for git
> continuity, but the project's *goal* — and this README — point at the atlas.

**Principal investigator:** Juan Pablo Chancay · jpcpol@gmail.com
**Started:** 2026-06-24 · **Target venue:** NeurIPS / ICLR (workshop)
**License:** see [LICENSE.md](LICENSE.md) — CC BY-NC 4.0 (docs/theory) + AGPL-3.0 (src)

---

## The arc in one sentence

The project began by asking *"can we quantize an LLM better by rotating its weights into the natural
Fisher basis (analogous to measuring in the Hamiltonian's eigenbasis)?"* — and ended up answering
*"that idea did not work, but refuting it revealed a new, reproducible geometric structure in
attention: each head lives in its own ~7D manifold, and these manifolds are mutually non-aligned
(overlap O_h ≈ 0.28), stably across model scale and corpus."*

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
  across scale and corpus.*
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
| [docs/paper_draft.md](docs/paper_draft.md) | **Complete preprint (frozen).** Central result: scale-invariant atlas (O_h ≈ 0.28). §0 ties the results back to the original objective; §3 orders the evidence by strength; §5 Related Work with references [1]–[16]. **Start here.** |
| [docs/paper_skeleton.md](docs/paper_skeleton.md) | Earlier paper skeleton (thermodynamic structure + intrinsic geometry). |
| [docs/research_questions.md](docs/research_questions.md) | Open research questions (Q01–Q06 and derivatives). |
| [docs/retrospective_vs_original_goal.md](docs/retrospective_vs_original_goal.md) | What the results mean against the founding hypothesis, with the outcome stated plainly. |
| [docs/figure_data.json](docs/figure_data.json) | Single source of truth for the figures (overlap matrices, dims, run scalars). |
| [docs/figures/](docs/figures/) | The paper's 8 figures (5 main + 3 supplementary). Fig 1 = H×H overlap matrix (the iconic one). |

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
| [src/autoencoder.py](src/autoencoder.py) | **EXP-Q06:** per-head nonlinear autoencoder (64→7→64) vs PCA rank-7. Clean negative: the manifold is not functionally compressible by this route. |
| [src/figure_data.py](src/figure_data.py) | Collects the figure data → `docs/figure_data.json`. |
| [src/make_figures.py](src/make_figures.py) | Renders the 8 figures → `docs/figures/`. |

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
| Scale-invariant atlas (O_h ≈ 0.28) | ✅ Central result, frozen |
| Per-head ~7D nonlinear manifold | ✅ |
| Functional compression of the manifold (Q06) | ❌ Honest negative (AE ≈ PCA) |
| Preprint | ✅ Complete draft, frozen |
| Bibliographic metadata confirmation | ⬜ Pending (reference manager) |
| Cross-architecture (Llama / Mistral) | ⬜ Pending — would promote "scale-invariant within GPT-2" |

---

## Future directions (prioritized, with NQP's caution)

1. **Geometric routing across heads** — dynamic activation of a subset of heads (MoE-like, but by
   latent geometry rather than learned logits). The most promising applied bet.
2. **Diagnostic metrics** — use O_h to detect head collapse/redundancy; requires no architectural
   change.
3. **Atlas stability under fine-tuning** — is the atlas more stable than the weights across
   pretraining → instruction tuning → RLHF?
4. **(Speculative)** structured compression (shared dictionary + per-head coordinates), geometric
   regularization, hierarchical macro/micro architectures.

The central medium-term question: **is the geometry causal or merely descriptive?**
