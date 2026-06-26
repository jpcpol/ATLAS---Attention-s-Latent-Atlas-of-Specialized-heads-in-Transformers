# NQP — What the results mean against the initial objective

**Date:** 2026-06-25 · **Researcher:** Juan Pablo Chancay

This document closes the project's arc: it contrasts what was obtained with the **founding
hypothesis** declared in `CLAUDE.md`, with the outcome stated plainly.

---

## 1. The initial objective (verbatim reminder)

> **NQP — Natural Quantization via State Preparation.** Optimize the quantization of LLMs via
> transformations that align the discrete grid with the natural geometry of weight space.
>
> **Intuition:** in quantum mechanics a system collapses to an eigenvalue of the measurement
> operator; discreteness *emerges* from the geometry. In standard ML quantization the grid is
> *imposed* externally and generates noise because it does not respect that geometry.
>
> **NQP hypothesis:** there exists a preparation operator P̂ (from the Fisher metric) such that
> quantizing in the basis of P̂ minimizes the error — analogous to measuring in the Hamiltonian's
> eigenbasis.

In one sentence: **can we quantize better by first rotating into a Fisher-derived "natural basis"?**

---

## 2. The direct answer to that question: **NO**

The central hypothesis (NQP-C1) was **empirically refuted**. In order:

- **Diagonal Fisher (EXP-001):** the P̂ basis collapses to the identity (P̂ = I) — no rotation,
  no "natural basis" distinct from the canonical one. Quantizing in that basis is identical to RTN.
- **Block-wise Fisher with real rotation (A-G4):** a genuine rotation P̂ ≠ I **does not beat** the
  strong baselines (GPTQ + AWQ + QuIP). The analogy with "measuring in the Hamiltonian's basis"
  turned out to be **metaphorical**: the activation Fisher is rank ~2, not the rich operator the
  quantum intuition presupposed.

**Conclusion about the declared objective:** NQP, *as a quantization method*, does not exist as an
advantage. The activation Fisher geometry is not the "Hamiltonian" that would justify a privileged
measurement basis. The project **did not produce** the deployment tool envisioned at the start.

---

## 3. Why the project did not end there: the pivot

The refutation left a finer question standing. If the quantum analogy fails in *quantization*, does
it fail entirely, or is there a part of the quantum structure that **does** describe the
transformer? That reoriented the project in two jumps:

1. **Weight/activation uncertainty principle (NQP-U).** Do the Fisher bases of weights and
   activations *fail to commute* (like incompatible observables)? **U1a: yes** (angle 48.8° vs 83°
   random — real incompatibility). **U1b: no operational consequence** (the angle↔error correlation
   is spurious: it drops to −0.04 when controlling for ε_W). A geometric truth, with no payoff in
   quantization.

2. **Thermodynamic/geometric characterization of attention** (the successor line). Here the anchor
   stopped being a metaphor: `softmax(QKᵀ/√d)` **is** literally a Boltzmann distribution. That exact
   identity opened the line that did produce positive results.

---

## 4. What the project DID discover (and was not looking for)

The project's central result is **not about quantization** but about **representation geometry**,
collected in the paper *"A Scale-Invariant Atlas of Head-Specific Manifolds in Transformer Residual
Attention"*:

| Finding | Status |
|---|---|
| The attention residual ε = Attn − V_{i\*} lives on a **~7D nonlinear manifold** per head (vs ~30 linear) | ✅ robust |
| Heads occupy **mutually non-aligned subspaces** (overlap O_h ≈ 0.28) | ✅ robust, **central result** |
| The non-alignment is **scale-invariant** (small/medium/large, CIs overlap) and **corpus-invariant** (WikiText vs C4) | ✅ robust |
| Thermodynamic phase structure (liquid→crystal) + S_vn as an **uncertainty proxy** | ✅ bridge result |
| Not compressible by hard selection (Top-k), by linear low-rank, or by a per-head nonlinear autoencoder | ✅ three clean negatives |

The "natural geometry of the space" that NQP sought **exists** — but not in the *weights* (where it
was sought for quantization), rather in the *attention residual*, and it **does not translate into
compression**.

---

## 5. The epistemological lesson (the project's real return)

The initial objective was a **bet on a physical analogy**. The project's value was not confirming
it but **measuring with discipline where the analogy is literal and where it is decorative**:

- **Literal and useful:** softmax = Boltzmann → measurable thermodynamics of attention.
- **Real but inert:** weight/activation non-commutativity (uncertainty with no consequence).
- **Decorative:** "Hamiltonian basis" for quantization (rank ~2 activation Fisher).

A recurring methodological pattern the project consolidated: **the naive tool over-concludes; the
correct control reframes.** It appeared four times — L2 error biases against NQP; bivariate vs
partial correlation in U1b; linear rank (PCA) vs intrinsic dimension (TwoNN); and "atlas" as a
formal fiber bundle vs a descriptive one. Each time, the honest control averted an overclaim.

---

## 6. Verdict against the initial objective

- **Was the declared objective (natural quantization via Fisher) met?** **No.** Refuted and
  documented, not quietly abandoned.
- **Did the project fail?** **No.** It turned a refuted physical hypothesis into a positive,
  reproducible, well-scoped interpretability result with negative controls — a paper.
- **Is the relation to CAL/L2 (deployment) that CLAUDE.md anticipated still alive?** Not via the
  intended route: NQP is not quantization infrastructure. Its contribution is conceptual (a
  geometric representation of attention), not deployment.

> NQP began by asking *how to measure better in order to discretize weights* and ended by answering
> *how attention is organized geometrically*. The original objective was refuted; the working method
> used to refute it produced the result that matters.
