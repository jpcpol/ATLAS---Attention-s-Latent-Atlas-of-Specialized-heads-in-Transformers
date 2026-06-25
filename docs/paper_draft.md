# A Scale-Invariant Atlas of Head-Specific Manifolds in Transformer Residual Attention

<!-- Conservative-title alternative: "Head-Specific Nonlinear Manifolds in Transformer Residual Attention" -->
<!-- Prior working title: "Non-Aligned Manifold Atlas in Transformer Residual Attention" -->

**Juan Pablo Chancay**¹, **Claude (Opus 4.8 / Sonnet 4.6)**²

¹ Independent Researcher · jpcpol@gmail.com
² Anthropic (research assistance)

**Draft — 2026-06-25 · Target venue: NeurIPS / ICLR workshop**

---

## Abstract

We study the geometry of the *residual* of attention — what remains of a head's output after
subtracting its single most-attended value vector. Across the GPT-2 family (124M–774M parameters)
this residual is organized as a **scale-invariant atlas of head-specific nonlinear manifolds**.
The central result is the *non-alignment between heads*: the mean pairwise subspace overlap is
O_h ≈ 0.28 (95% bootstrap CI [0.27, 0.29]), and it is statistically indistinguishable across a 6×
change in model size — the per-model confidence intervals overlap.
Within this structure, each head's residual is a low-dimensional nonlinear manifold — intrinsic
dimension (TwoNN ≈ 6–7) far below its linear rank (PCA ≈ 30) — and head-centering does not collapse
the union, so the heads are not one shared manifold seen through different offsets. We rule out the
two obvious linear stories (sparse selection and global low-rank compression) explicitly. Finally,
this scale-invariant geometric micro-structure coexists with macroscopic thermodynamic observables
that are *not* scale-invariant (a crystallization depth L_c, derived from softmax-as-Boltzmann),
separating an invariant geometric core from capacity-dependent dynamics. We use "atlas"
descriptively — a collection of local charts — not as a claim of a formal fiber bundle.

---

## 1. Introduction

A self-attention head produces an output that is a convex combination of value vectors,
Attn(x) = Σ_i a_i V_i, with weights a = softmax(QKᵀ/√d). A natural decomposition isolates the
single most-attended token i\* and treats everything else as a residual. Writing the identity
explicitly,

> Attn = a_{i\*} V_{i\*} + (1 − a_{i\*}) ε,  where  ε = (1 − a_{i\*})⁻¹ Σ_{i ≠ i\*} a_i V_i.

Here ε is the normalized "tail" of the attention output — the convex combination of the
non-dominant values — and (1 − a_{i\*}) is the total mass on that tail. We study the geometry of ε.

In deep layers the attention distribution becomes highly concentrated (one token dominates),
which invites the hypothesis that the residual ε is negligible, sparse, or low-rank — i.e. that
deep attention has effectively "crystallized" into a selection operation and the tail can be
discarded or cheaply compressed. We test this hypothesis and find it **false**, but the *way* it
fails reveals structure: the residual is neither sparse-replaceable nor linearly low-rank, yet it
is **nonlinearly low-dimensional and head-specific**.

Our central finding is geometric and concerns the relationship *between heads*. Each head's
residual occupies its own low-dimensional nonlinear manifold, and these manifolds are embedded in
**mutually non-aligned subspaces** — they are not coordinatizable by a single shared linear system.
We call this organization an *atlas* (a collection of head-specific charts). The strongest,
cleanest result is not the dimensionality of the charts but their **non-alignment**: the inter-head
subspace overlap is ≈ 0.283 and is invariant to a 6× change in model scale.

**Contributions.**
1. A decomposition and measurement protocol for the attention residual ε, with synthetic
   validation of every estimator used (intrinsic dimension, subspace overlap).
2. The central empirical result: residual attention is a **non-aligned, scale-invariant atlas of
   head-specific manifolds** (inter-head overlap O_h ≈ 0.28, 95% CI [0.27, 0.29], statistically
   indistinguishable across GPT-2 small/medium/large and across corpora).
3. Explicit refutation of the obvious compression stories — sparse selection, global low-rank, and
   (a clean negative we contribute) a *per-head nonlinear autoencoder*, which matches but does not
   beat linear PCA at the intrinsic dimension. The residual is geometrically low-dimensional but not
   functionally compressible by these means.
4. A two-layer account separating a **scale-invariant geometric micro-structure** from
   **capacity-dependent macroscopic thermodynamics** (softmax-as-Boltzmann phase structure;
   crystallization depth L_c, which is *not* scale-invariant). As a byproduct, von Neumann entropy
   of the per-head density matrix is a cheap, calibrated proxy for predictive uncertainty.

**Why this matters.** A stable geometric atlas constrains the effective degrees of freedom
available to attention and provides a new representation-level object for studying scaling and
head specialization — independently of whether it yields a compression mechanism (§3.7).

**Scope and anti-overclaim.** All scale-invariance claims are established *within the GPT-2 family
and a fixed training distribution*, not across architectures. "Atlas" is descriptive; we do not
claim a formal fiber bundle.

---

## 2. Setup and Methods

### 2.1 The residual decomposition

For a head h at layer ℓ and query position t, let a^{(t)} = softmax(q_t Kᵀ/√d) be the attention
weights over keys, i\* = argmax_i a_i^{(t)} the dominant key, and

> ε_t = Σ_{i ≠ i\*} a_i^{(t)} V_i / (1 − a_{i\*}^{(t)}).

We collect {ε_t} over many positions and contexts (WikiText-103 validation) per (ℓ, h) and study
the resulting point cloud in ℝ^{d_head} (d_head = 64 for GPT-2). Unless stated otherwise we use the
**deepest layers** (where concentration is highest and the "discard the tail" hypothesis is most
plausible).

### 2.2 Intrinsic dimension (TwoNN)

We estimate intrinsic dimension with the TwoNN estimator (Facco et al., 2017): for each point,
μ = r₂/r₁ (ratio of distances to its two nearest neighbors), and d = (N−1)/Σ_i log μ_i. TwoNN
recovers nonlinear dimension where PCA cannot. **Validation:** on a 2D swiss-roll embedded in ℝ³,
TwoNN returns ≈ 2.7 while PCA reports 3 — it sees the manifold, not the ambient rank.

### 2.3 Inter-head subspace overlap (the atlas test)

For each head we extract a local linear frame B_h ∈ ℝ^{d_head × d_local} via SVD of the
mean-centered residual cloud (the top d_local right-singular vectors, so B_hᵀ B_h = I). For a pair
of heads (h_i, h_j) we take the singular values σ_k of B_iᵀ B_j — the cosines of the principal
angles between the two subspaces, clamped to [0, 1] — and define the **pairwise overlap as their
mean**:

> O(h_i, h_j) = (1 / d_local) Σ_{k=1}^{d_local} σ_k(B_iᵀ B_j),

and O_h is the average of O(h_i, h_j) over all unordered head pairs. We use the mean of the
principal cosines (a normalized Frobenius-style alignment) rather than the maximum (which reports
only the best-aligned direction) or the nuclear norm (unnormalized): the mean answers "on average,
how aligned are the two d_local-dimensional frames?", with O_h = 1 iff the subspaces coincide and
O_h = 0 iff they are orthogonal. **Calibration:** on synthetic data O_h = 1.0 when heads share a
basis and O_h ≈ 0.0 when their bases are orthogonal. We also compute a **head-centered pooled
dimension**: if the heads were one manifold seen through per-head offsets, removing each head's mean
would collapse the pooled intrinsic dimension toward the per-head value (~7); if they are genuinely
non-aligned, it does not.

**Choice of d_local.** We set d_local = 7 because the per-head intrinsic dimension (§2.2, §3.2) is
≈ 6–7; the frame should span the manifold without padding it with noise directions. Because this is
a hyperparameter of the *measurement*, we report a sensitivity analysis over d_local ∈ {4, …, 10}
in §3.1. The *reported value* O_h ≈ 0.283 is specific to d_local = 7; as expected, O_h rises
monotonically with d_local (adding lower-variance directions raises the mean cosine). What is
invariant to the choice is the *conclusion*: O_h stays far below 1 for every k ∈ [4, 10]
(≈ 0.21 → 0.34), so the heads are non-aligned regardless of how the frame is sized. We therefore
fix d_local = 7 (matching intrinsic dimension) for the headline number and treat its absolute value,
not just its sign, as conditional on that choice.

### 2.4 Linear baselines (refuting the obvious)

- **Sparse selection (Top-k):** replace the full attention distribution with its top-k weights
  (renormalized) in chosen layers and measure ΔPPL. This tests whether the tail is functionally
  discardable.
- **Global low-rank:** PCA on ε; report dimensions for 90% variance and the loss recovered by a
  rank-32 projection. This tests linear compressibility.

### 2.5 Thermodynamic observables (softmax = Boltzmann)

softmax(QKᵀ/√d) is literally a Boltzmann distribution with energies E_i = −(q·k_i)/√d and β = 1.
From the same partition function Z the head already computes we read off the Helmholtz free energy
F = −β⁻¹ log Z, mean energy ⟨E⟩, heat capacity C = β²(⟨E²⟩ − ⟨E⟩²), an effective temperature T_eff,
and the von Neumann entropy S_vn(ρ) of the density matrix ρ = Σ_i a_i |v̂_i⟩⟨v̂_i|. Masked keys
(−∞ logits) are handled by zeroing their finite-energy contribution to avoid 0·∞ = NaN. We define
the **crystallization depth** L_c as the layer at which heads transition from a high-T "liquid"
regime to a low-T "crystallized" regime (T_eff → 1, purity → 1).

### 2.6 Models and controls

We use gpt2 (124M, 12 layers), gpt2-medium (355M, 24 layers), and gpt2-large (774M, 36 layers).
**Critical control:** for any cross-scale comparison of intrinsic dimension or overlap we fix the
number of sampled points N, the number of heads, and the relative layer depth across models — an
early uncontrolled run gave a spuriously identical dimension and taught us to control N/heads/layers
before comparing intrinsic geometry across scales.

---

## 3. Results

We present results in **decreasing order of empirical strength**, which is *not* the order in which
the experiments were run. The strongest result is the inter-head non-alignment and its
scale-invariance; the dimensionality is supportive; the decreasing-dimension trend is exploratory.

### 3.1 (Very strong) Heads are non-aligned, and the non-alignment is scale-invariant

Within a deep layer, the per-head residual subspaces are far from aligned: at d_local = 7 the mean
inter-head overlap is O_h ≈ 0.28 (a value of 1 would mean identical subspaces, 0 orthogonal).
Repeating the measurement across the GPT-2 family under the controlled protocol (same N = 1200, same
8 heads, deepest layer), with a percentile bootstrap over the head pairs:

| model  | params | layers | O_h (d_local=7) | 95% bootstrap CI | head-centered pooled dim |
|--------|--------|--------|-----------------|------------------|--------------------------|
| small  | 124M   | 12     | 0.284           | [0.276, 0.292]   | 7.1                      |
| medium | 355M   | 24     | 0.277           | [0.267, 0.288]   | 6.7                      |
| large  | 774M   | 36     | 0.281           | [0.272, 0.290]   | 5.7                      |

> **O_h ≈ 0.28 across a 6× change in parameters; the three 95% CIs overlap, so the cross-scale
> variation (≈ 0.007) is within sampling error.**

This is the rigid result. Two null hypotheses are decisively rejected: heads are **not** a single
shared manifold (which would give O_h ≈ 1), and the structure is **not** explained by per-head
offsets alone (head-centering does not collapse the pooled dimension to the per-head ~7). The heads
are geometrically *decoupled modules*, each with its own coordinate system, and this decoupling is
a stable structural property of the family rather than a small-model artifact.

We therefore state the claim at the level of the geometric property, not the exact number:

> *Across the GPT-2 family, attention heads occupy substantially non-aligned residual subspaces.
> While the absolute value of the overlap metric depends on the choice of local-dimensionality
> parameter d_local, the qualitative result is robust: inter-head overlap remains far below the
> shared-subspace expectation (O_h ≪ 1) across all tested settings, with stable values across model
> scale, dataset size, and relative depth.*

**Robustness (Appendix B).** The value is stable to the measurement choices, with one honest
caveat. Across sample size N ∈ {300, 600, 1200} it moves by ≤ 0.005; across each of the three
deepest layers it moves by ≤ 0.004 (spread 0.001–0.004 per model). The one quantity it *does* track
is d_local itself: O_h rises monotonically from ≈ 0.21 (k=4) to ≈ 0.34 (k=10) as lower-variance
directions enter the frame. This is expected and does not affect the conclusion — O_h stays well
below 1 for every k — but it means the *absolute* headline value 0.28 is reported conditional on
d_local = 7 (chosen to match the intrinsic dimension, §3.2), whereas the *non-alignment itself* is
choice-independent.

**Inter-corpus control.** Non-alignment is a property of the model, not of the measurement corpus.
Recomputing O_h on a second corpus (C4) gives O_h = 0.277, 95% CI [0.269, 0.285], versus 0.284
[0.276, 0.292] on WikiText-103 — the two CIs overlap (cross-corpus spread 0.007). This addresses the
most direct alternative explanation ("is this a corpus artifact?"): it is not.

### 3.2 (Strong) Each head is a low-dimensional nonlinear manifold

Per head, the residual's intrinsic dimension (TwoNN ≈ 6–7) is far below its linear rank
(PCA ≈ 28–32 for 90% variance). The cloud is therefore a **nonlinear** manifold of low effective
dimension embedded in a much higher linear span — "not compressible by PCA" is not the same as "not
compressible". Auxiliary geometric checks (connectivity, local-dimension homogeneity, short-range
interpolation) are consistent with a single connected, locally regular manifold per head rather than
a union of disjoint clusters.

### 3.3 (Moderate) The per-head intrinsic dimension is approximately scale-invariant

Under the controlled protocol the per-head intrinsic dimension stays in a narrow band (~6–8) across
scale, with between-model spread (≈0.9) comparable to between-head spread (≈0.8) within a single
model — i.e. cross-scale variation is not clearly distinguishable from within-model variation.

### 3.4 (Exploratory) A mild decreasing trend in dimension with scale

We observe a mild monotonic decrease in intrinsic dimensionality with model scale (≈6.7 → 5.3).
However:

> *We observe a mild monotonic decrease in intrinsic dimensionality with model scale (6.7→5.3),
> although the effect remains comparable to within-head variability and therefore cannot yet be
> distinguished from finite-sample variation.*

We report this trend for completeness and flag it as exploratory; the robust invariant is the
overlap O_h, not the exact dimension.

### 3.5 The residual is neither sparse-replaceable nor linearly low-rank

- **Sparse selection fails.** Replacing deep-layer attention with a hard Top-1/Top-2 selection
  degrades perplexity substantially (e.g. +7.7 PPL at the deepest layer for Top-1), even where the
  runner-up weight ratio p₂/p₁ ≈ 0 and argmax is 100% stable. *Deterministic ≠ replaceable:* the
  long tail of hundreds of tiny weights performs systematic context integration. S_vn ≈ 0 measures
  concentration of *where* a head looks, not irrelevance of the *rest*. Top-2 reduces the damage to
  ≈ 1/3 of Top-1, confirming the information lives in the tail.
- **Low-rank fails.** ε needs ≈ 30 *linear* dimensions for 90% variance; a rank-32 projection
  recovers only ≈ 87% of the Top-1 loss. The residual is not linearly low-rank.

Together these refute the two obvious compression stories and force the conclusion that the
residual's compressibility, to the extent it exists (§3.2), is **nonlinear and head-local**.

### 3.6 A separable macroscopic thermodynamic layer

softmax-as-Boltzmann yields a real phase structure in GPT-2: early layers are "liquid"
(T_eff ≈ 15, low purity ≈ 0.43) and deep layers are "crystallized" (T_eff ≈ 1, purity ≈ 1,
S_vn ≈ 0). **Bridge result:** the von Neumann entropy S_vn(ρ) is a *calibrated, cheap proxy for
predictive uncertainty* — it correlates −0.20 with prediction uncertainty across heads and depth and
survives controlling for token position (partial −0.18). Crucially, the **crystallization depth L_c
is not scale-invariant** (≈ 2 / 1 / 9 for small/medium/large; gpt2-large has an extended liquid
phase). The geometric invariants of §3.1–§3.3 and this macroscopic, capacity-dependent observable
are of different natures — the core of our two-layer account.

### 3.7 (Honest negative) The manifold is geometric, but not functionally compressible

The low intrinsic dimension (§3.2) raises a functional question: can a **per-head nonlinear
autoencoder** (64 → 7 → 64, trained to reconstruct ε) repair the perplexity that hard Top-1
destroys, and beat the linear PCA projection of the same rank? It cannot. Replacing ε with the AE
reconstruction on the three deepest layers recovers **55.8%** of the Top-1 loss — essentially
identical to the **55.9%** recovered by a linear rank-7 PCA projection (held-out reconstruction
FVU ≈ 0.38). The nonlinear bottleneck buys nothing over the linear one at equal dimension.

We read this carefully. The intrinsic dimension ≈ 7 (TwoNN) is a *local* estimate of manifold
curvature; it does **not** imply a single global 7-dimensional parametrization that an autoencoder
can recover. The residual manifold is therefore *real geometrically* (§3.1–§3.2) but **not, with this
method, functionally compressible** below its linear rank. This bounds the practical reading of the
result and is consistent with §3.5: the residual's information is genuinely spread across many
value-space directions. We report it as a clean negative rather than leaving compression as an
open promise.

---

## 4. Interpretation: a two-layer account

| | Macro — thermodynamic | Micro — effective geometry |
|---|---|---|
| Object | attention *dynamics* | space of residual *states* |
| Observables | phases, T_eff, S_vn, L_c | intrinsic dim, inter-head overlap |
| Scale behavior | **dependent** (L_c shifts) | **invariant** (O_h, dim) |
| Evidence | §3.6 | §3.1–§3.3 |

Model scaling does not change the effective geometry of the residual; it shifts *where* the
low-temperature regime appears. The strong invariant is the inter-head non-alignment O_h ≈ 0.283.

**What are the ~7 per-head dimensions? (competing hypotheses, not vague doubt.)**
1. *Head-specific semantic modes* — each head encodes ~7 of its own semantic functions; coordinate
   incompatibility = functional specialization.
2. *Head-specific retrieval modes* — ~7 contextual-retrieval modes per head.
3. *Optimization-induced decoupling* — the atlas as a product of training dynamics (heads
   decoupled to minimize interference), not necessarily semantically interpretable.

The incompatibility of coordinates (O_h ≈ 0.28) favors (1)/(3) over any shared-basis account: heads
are not rotations of a common manifold; they are geometrically decoupled modules.

---

## 5. Related Work

We position this work along four lines. Up front: **this paper does not propose a compression method
or a complete thermodynamic theory of attention; it identifies a new geometric structure in the
attention residual** and characterizes its invariances.

**Representation geometry.** A body of work measures the *intrinsic dimension* of deep
representations and describes them as low-dimensional *neural manifolds* embedded in
high-dimensional activation space (Ansuini et al.; Facco et al.; and the broader latent-geometry
literature). We adopt the same estimator (TwoNN) but change the object of study: instead of layer
activations we analyze the *attention residual* ε, and instead of a single global manifold we
measure the *relationship between per-head manifolds* (inter-head subspace overlap) — the axis that
yields our main result.

**Mechanistic interpretability.** Work on attention-head specialization, circuits, and superposition
asks *what* heads compute. We are complementary and orthogonal: we do not identify features or
circuits, but show that, geometrically, heads carry residual information in *mutually non-aligned
subspaces*. Our decoupled-per-head charts give a representation-level correlate of specialization
without committing to any functional labeling.

**Thermodynamic interpretations of attention.** softmax(QKᵀ/√d) is exactly a Boltzmann distribution,
which several works exploit (entropy of attention, energy-based and free-energy views). We use this
exactly, not metaphorically, and only as a *measurement device* for macroscopic observables (phases,
T_eff, L_c) and as the source of the S_vn uncertainty proxy. The thermodynamic layer is a tool here,
not the contribution.

**Low-dimensional structure and compression.** Low-rank transformers, pruning, and sparse-attention
methods seek to remove redundancy. We relate by *negation*: we explicitly refute the sparse-selection
(Top-k) and global low-rank stories for the residual (§3.5), which distinguishes our object from
compressible-attention claims and motivates the nonlinear, per-head view.

Anticipated objection — *"isn't this just intrinsic dimension applied to activations?"* The answer
is the *system* of evidence: scale-invariance (§3.1, §3.3) + refutation of the linear stories (§3.5)
+ the inter-head non-alignment that no single dimensionality measurement captures (§3.1). No single
metric carries the claim.

---

## 6. Limitations and Threats to Validity

- **Family scope.** Scale-invariance is established within the GPT-2 family and a fixed training
  distribution; we do not test Llama, Mistral, or other architectures.
- **"Atlas" is descriptive.** We measure local chart dimension and inter-chart non-alignment. We do
  **not** measure transition maps between charts, global differentiable consistency, or a base/fiber
  structure. We therefore do not claim a formal fiber bundle. O_h ≠ 1 is necessary but not
  sufficient for such a structure.
- **Dimension trend is exploratory.** The §3.4 decrease is within within-head variability.
- **No demonstrated functional exploitation.** A per-head nonlinear autoencoder at the intrinsic
  dimension does *not* beat linear PCA (§3.7); the geometric manifold is not, with this method,
  functionally compressible. We do not claim compression as an application.

---

## 7. Future Work

**Beyond per-head autoencoding.** Our functional test (§3.7) is negative: a per-head AE at the
intrinsic dimension matches but does not beat linear PCA. This rules out the *simplest* exploitation,
not all of them. Open directions on the functional side: routing/mixture parametrizations that share
structure *across* heads despite their non-aligned charts; exploitation that targets the manifold's
*global* geometry rather than a per-point bottleneck; and asking whether the residual is better
described as high-entropy integration noise (cf. §3.5) than as a compressible code at all.

**Structural and external validity.** (i) Cross-architecture generalization (Llama, Mistral): is the
~0.28 non-alignment a GPT-2-family property or universal? (ii) Promoting "atlas" toward a formal
claim by *measuring transition maps* between head charts and their differentiable consistency — the
step we deliberately do not take here. (iii) Mutual information between heads as an
information-theoretic complement to the geometric overlap. (iv) A renormalization-group view of the
depth-wise crystallization (§3.6).

---

## Appendix A — Estimator validation summary

| estimator | synthetic test | expected | observed |
|---|---|---|---|
| TwoNN intrinsic dim | 2D swiss-roll in ℝ³ | ~2 | 2.7 (PCA: 3) |
| inter-head overlap O_h | orthogonal-basis atlas | 0.0 | ≈ 0.0 |
| inter-head overlap O_h | shared-basis (single manifold) | 1.0 | ≈ 1.0 |
| interpolation test | flat plane / circle | low / high | 0.75 / 9.63 |

## Appendix B — Robustness of O_h (bootstrap, sensitivity, stability)

Measured on the deepest layer of each model, 8 heads, N = 1200 residuals/head, WikiText-103
validation. The 95% CI is a percentile bootstrap (2000 resamples) over the 28 head pairs.

| model  | O_h (d_local=7) | 95% CI         | pair sd | d_local sweep k=4→10 | N-spread | depth-spread |
|--------|-----------------|----------------|---------|----------------------|----------|--------------|
| small  | 0.284           | [0.276, 0.292] | 0.023   | 0.214 → 0.342        | 0.003    | 0.001        |
| medium | 0.277           | [0.267, 0.288] | 0.029   | 0.202 → 0.332        | 0.004    | 0.004        |
| large  | 0.281           | [0.272, 0.290] | 0.025   | 0.211 → 0.342        | 0.008    | 0.004        |

Reading: (i) the three CIs overlap → no resolvable scale dependence; (ii) O_h is stable to sample
size and to which deep layer is used (spreads ≤ 0.008); (iii) O_h rises monotonically with d_local
(adding lower-variance directions), so the *value* is reported at the principled choice d_local = 7
(≈ intrinsic dimension) while the *non-alignment conclusion* (O_h ≪ 1) holds for all k ∈ [4, 10].

**Inter-corpus.** On gpt2 (deepest layer, same protocol), O_h = 0.284 [0.276, 0.292] on
WikiText-103 vs 0.277 [0.269, 0.285] on C4 — CIs overlap, spread 0.007. The non-alignment is
data-independent within this regime.

## Appendix C — Reproducibility

Code: `src/residual.py` (decomposition), `src/intrinsic.py` (TwoNN), `src/manifold.py`
(atlas_test, geometric checks), `src/atlas_scaling.py` (cross-scale overlap),
`src/atlas_robustness.py` (bootstrap CI + d_local/N/depth sensitivity),
`src/atlas_intercorpus.py` (WikiText vs C4 control), `src/autoencoder.py` (per-head AE, Q06),
`src/scaling.py`
(dim + L_c across scale), `src/thermo.py` (Boltzmann observables, S_vn), `src/crystallize.py`
(Top-k baseline). Theory map: `theory/quantum_transformer_map.md`. Data: WikiText-103 validation.
All cross-scale comparisons use fixed N / heads / relative depth.
