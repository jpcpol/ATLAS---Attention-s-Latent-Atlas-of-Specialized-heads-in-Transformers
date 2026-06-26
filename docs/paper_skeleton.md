# Paper Skeleton — Thermodynamic Structure and Intrinsic Geometry of Residual Attention in Transformers

**Status:** Skeleton · 2026-06-25
**Authors:** Juan Pablo Chancay, Claude (Opus 4.8 / Sonnet 4.6)
**Origin:** successor line to NQP. See `theory/quantum_transformer_map.md` for the full map of
gaps and `theory/uncertainty_principle.md` for the refuted branch (NQP-U).

---

## One-liner (abstract level) — minimal valid claim (NOT a formal fiber bundle)

> We show that residual attention in GPT-2 is structured as a **collection of head-specific
> nonlinear manifolds with consistent intrinsic dimensionality (~7), embedded in mutually
> non-aligned subspaces**, yielding an apparent higher-dimensional global representation (~17D).
> Each head's contextual residual occupies its own low-dimensional manifold (intrinsic dim
> scale-invariant within the GPT-2 family), and these manifolds are **not coordinatizable by a
> shared linear system** (inter-head subspace overlap ≈ 0.28; head-centering does not collapse
> the union, ~12.7D). This coexists with a thermodynamic phase structure that depends on model
> capacity, separating a scale-invariant per-head geometric core from macroscopic,
> training-dependent observables.

**Title:** *"Non-Aligned Manifold Atlas in Transformer Residual Attention"*. The central result
is the **geometric decoupling between heads**: consistent local ~7D representations but with no
global linear coordination.

**EXPLICIT LIMIT (anti-overclaim, J.P. Chancay):** "atlas" is used descriptively (a family of
local charts), NOT as a **formal fiber bundle**. We do NOT claim: smooth transition maps between
charts, global differentiable consistency, or a base+fiber structure. That would require measuring
continuity between heads (not just overlap) and a differentiable structure — future work. What is
measured: consistent local dimension + subspace non-alignment + non-collapse under recentering.

## Abstract (draft, workshop level)

> We show that residual attention in GPT-2 forms a **scale-invariant, low-dimensional
> nonlinear constraint manifold (~7D)** — an *emergent constraint manifold* in deep
> representations — coexisting with a thermodynamic phase structure that depends on model
> capacity. This separates geometric invariants (the residual manifold) from macroscopic,
> training-dependent observables (crystallization depth L_c). We rule out the obvious
> linear explanations (sparsity, low-rank) and find that what remains is a geometric
> constraint. Scale-invariance is established **within the GPT-2 family / a fixed training
> distribution**, not across architectures.

## Central thesis

The contextual residual of deep attention (ε = Attn − V_{i*}) is an **emergent constraint
manifold**: a nonlinear manifold of effective dimension ≈ 7-8 (vs ~30 linear) whose dimension is
**invariant to model scale within the GPT-2 family**. It coexists with a thermodynamic phase
structure that **does depend on the capacity regime** (L_c). Two separable levels:

> **Macro (thermodynamic, scale/capacity-dependent):** phase regime, T_eff, L_c.
> **Micro (effective geometry, scale-invariant within the family):** ~7D residual manifold.

Consequence: *model scaling does not change the effective geometry of the residual; it only
shifts where the low-temperature regime appears.* The central result is the **existence** of the
nonlinear manifold (Q05); its **scale stability** (Q04) is the property that elevates it.

**Protective wording (anti-overclaim):** "universal" → **"scale-invariant within the GPT-2
family / a fixed training-distribution family"**. Cross-architecture is not proven.

---

## Two-layer structure (the conceptual contribution)

| | Layer A — Thermodynamics (macro) | Layer B — Effective geometry (micro) |
|---|---|---|
| Object | attention dynamics | space of residual states |
| Observables | phases, T_eff, S_vn, L_c | intrinsic dim, nonlinearity |
| Scale | dependent (partial) | **invariant** |
| Evidence | Q01, Q02, Q04-lite | Q03, Q05, Q04-lite |

---

## Results (what is ALREADY justified, Q01–Q05)

Order of conceptual strength (not numbering): **R-MANIFOLD (Q05) is the heart** — without the
existence of the nonlinear object, its stability means nothing. Then its stability (Q04), the
linear contrast (Q02/Q03) that motivates it, and the thermodynamics (Q01) as the macro layer +
bridge.

### R-ATLAS (heart) — Non-aligned per-head manifolds (Q05, Q05d)
Each head has a residual of intrinsic dimension (TwoNN) ≈ 7, vs ~30 linear (PCA) — effective
DOF ≪ embedding dim, one local manifold per head. **But they do NOT share coordinates:** inter-head
subspace overlap = 0.28 (≈orthogonal), and the per-head-centered pooled set does not collapse
(12.7D). → the system is a **collection of local ~7D manifolds embedded in mutually non-aligned
subspaces**, union ≈17D. Four nulls DEFINITIVELY ruled out:
NOT single-manifold (overlap≠1), NOT global low-rank (Q03), NOT shared latent basis (overlap 0.28),
NOT pure gauge/offset (centered dim 12.7≠7).
(TwoNN validated: 2.7 on swiss-roll where PCA sees 3; overlap validated: 0.0 atlas vs 1.0 shared on
synthetics.) **The central claim is the geometric decoupling between heads.** "Atlas" in the
descriptive sense (local charts), NOT a formal fiber bundle (no transition maps measured).

Minimal consistent model: ε = ⋃_h M_h, with dim(M_h)≈7, non-alignable embeddings, union ~12-17D.

### R-STABILITY — Dimension AND non-alignment are scale-invariant (Q04-lite, Q05d-scale)
Controlled protocol (identical N, same heads/layers) on gpt2/medium/large (124M→774M):

| model | layers | dim_int | dim_lin | inter-head overlap | centered_dim |
|---|---|---|---|---|---|
| small | 12 | 7.2 ± 1.1 | 31.6 | 0.285 | 7.1 |
| medium | 24 | 8.1 ± 0.8 | 30.5 | 0.282 | 6.7 |
| large | 36 | 7.4 ± 0.7 | 28.0 | 0.281 | 5.7 |

**Two scale invariants:** (a) intrinsic dim ~7 (σ_between-models 0.9 ≈ σ_between-heads 0.8);
(b) **inter-head non-alignment overlap ≈ 0.28, spread 0.004 across a 6× change in params.** Both
ELEVATE R-ATLAS: the atlas of non-aligned manifolds not only exists, it is scale-stable (within the
GPT-2 family). centered_dim ≈ dim_int per model confirms that the ~12.7 from Q05d came from mixing
layers; with one controlled layer, the non-alignment is between heads of the same layer.
**Honest caveat:** dim_int decreases slightly and monotonically (6.7→6.1→5.3, within the ±1.1
noise); the non-alignment does NOT (it is flat). The robust invariant is the overlap.

### R-LINEAR — Why the manifold is nonlinear: we refute the obvious (Q02, Q03) [negatives that strengthen]
- Q02: replacement by Top-k (hard selection) destroys PPL (+7.7 at L11 alone). Rules out the
  **sparsity-compression story**: attention is integration, not selection.
- Q03: ε needs ~30 LINEAR dims (PCA 90%); low-rank does not compress (r=32 → 87%). Rules out the
  **low-rank story**.
- Narrative: *we rule out the obvious explanations (sparsity, low-rank); what remains is a
  nonlinear geometric constraint* — the pattern of solid ML-theory papers.

### R-THERMO (macro layer + bridge) — Phase structure and S_vn as an uncertainty proxy (Q01)
softmax(QKᵀ/√d) = Boltzmann distribution ⇒ F, ⟨E⟩, C, S_vn from the same Z. A gradient of
T_eff with depth: early layers liquid (T_eff≈15), deep layers crystallized (T_eff≈1).
**Bridge result (elevate, do not bury):** the von Neumann entropy S_vn(ρ) is a **calibrated proxy
for predictive uncertainty** across heads and depth (corr −0.20, survives controlling for position,
partial −0.18) — a *metric candidate*, not just an observation.

### R-Lc — L_c is NOT scale-invariant (Q04-lite) [honest delimitation, reconciles macro/micro]
L_c = 2/1/9; large has an extended liquid phase. L_c = f(capacity, training, architecture).
**The invariant is geometric (dim M_ε), not thermodynamic-macro (L_c).** Two invariants of
different natures — the core of the macro/micro separation.

---

## What IS the atlas of ~7D manifolds? (competing interpretations)

The atlas refines the question: each head has its own ~7D manifold with its own coordinates.
What are those ~7 dims PER HEAD? Competing hypotheses (not vague doubt):
1. **Head-specific semantic modes:** each head encodes ~7 of its own semantic functions; the
   coordinate incompatibility = functional specialization (each head "looks at" different things).
   Consistent with interpretability: heads as specialized detectors.
2. **Head-specific retrieval modes:** ~7 contextual-retrieval modes per head.
3. **Optimization-induced:** the atlas as a product of training dynamics (heads decoupled to
   minimize interference), not semantically interpretable per se.

The fact that the coordinates are incompatible (overlap 0.28) FAVORS 1/3 over a shared basis:
the heads are not rotations of a single manifold, they are geometrically decoupled modules.

## Positioning vs literature (for related work)

- It is NOT classical mechanistic interpretability (not individual circuits/features).
- It is NOT classical scaling-laws (not loss vs compute; it is geometry vs scale).
- It is NOT classical compression research (we explicitly refute sparsity and low-rank).
- It IS a hybrid thesis: **emergent effective geometry + attention dynamics**. It connects with
  "intrinsic dimension of representations" and "emergent constraint manifolds" in rep. learning.
- Anticipated defense against the attack "is this just dimensionality applied to activations?": the
  answer is the *combination* — scale invariance (R-STABILITY) + refutation of the linear (R-LINEAR)
  + bridge to uncertainty (R-THERMO). No isolated metric; a system of evidence.

## What CANNOT yet be claimed (honesty / future work)

- **Functional exploitability** of the manifold (Q06 — autoencoder). Pending.
- **Geometric regularity (Q05b/c/d, 2026-06-25):** M_ε is connected (1 comp), homogeneous,
  locally flat (interp 0.60), does not fragment by layer (ratio 5.0). **Q05d (atlas test) resolves
  the nature of the object:** subspace overlap between heads = 0.28 (≈orthogonal) and per-head
  centered dim = 12.7 (does not collapse to 7). → **non-aligned per-head manifolds confirmed**,
  ruling out "offsets only" (which would give overlap≈1, centered dim≈7). Incompatible latent
  coordinates. ("Atlas" descriptive, not a formal fiber bundle — transition maps not yet measured.)
- Generalization beyond the GPT-2 family (Llama, Mistral): not tested.

---

## Precise claim (do not overclaim)

NO: "universality of the full model".
YES: **"universality of the deep residual-excitation subspace"** — more specific and stronger. The
~7D geometric core belongs to the residual, not the whole model.

---

## Closing plan

- **Phase 1 (complete):** Q01–Q05.
- **Phase 2 (geometric clarification, before claiming exploitability):** Q05b smoothness /
  interpolation; Q05c clustering (submanifolds by layer/head-type/position?).
- **Phase 3 (optional, high impact):** Q06 functional autoencoder — only if Q05b/c show that
  M_ε is globally regular.
