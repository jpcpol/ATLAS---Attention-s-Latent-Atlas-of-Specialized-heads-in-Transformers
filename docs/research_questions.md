# NQP — Open research questions

**Last updated:** 2026-06-24

---

## RQ-1 (Central)
Does there exist a preparation operator $\hat{P}$ derived from the Fisher metric of an LLM
such that quantizing in the basis of $\hat{P}$ minimizes the quantization error relative to
any fixed-grid quantization method?

**Hypothesis:** Yes. The Fisher basis is the model's "natural" basis in the sense that, in it,
the weight directions are independent under the loss — and quantizing independent directions
with a bit budget proportional to their curvature is optimal (a rate-distortion argument).

---

## RQ-2
Is the Fisher matrix $F$ computable efficiently enough for models at practical scale
(7B–70B parameters)?

**State of the art:** K-FAC (Kronecker-factored approximation), diagonal Fisher, per-layer
block Fisher. The question is whether any approximation preserves enough structure for the
$\hat{P}$ derived from it to beat a random rotation (QuIP).

---

## RQ-3
Is there a notion of an "uncertainty principle" in weight space?

**Intuition:** if $\hat{P}$ diagonalizes $F$, there may be directions where weight precision
and activation precision cannot be optimized simultaneously — analogous to Heisenberg's
uncertainty principle between position and momentum.

**Tentative formalization:** for $\hat{P}$ that diagonalizes $F_W$ (Fisher with respect to
weights) and $G_A$ (Fisher with respect to activations), if $[\hat{P}_W, \hat{P}_A] \neq 0$,
then there is a fundamental trade-off between weight quantization error and activation error.

---

## RQ-4
Can NQP-C2 (the strong form of the conjecture) be true?

The conjecture says that quantizing in the natural basis with $b$ bits can *beat* FP32. This
would imply that $\hat{P}$ acts as a regularizer: by removing the low-curvature components
(barely relevant to the loss), it reduces overfitting to training noise.

**Analogy:** quantizing in the natural basis would be equivalent to truncating the small
Fisher eigenvalues — similar to low-rank approximation or structured dropout.

---

## RQ-5
How does NQP relate to existing methods?

- GPTQ uses the per-layer output Hessian → a local approximation of $F$
- QuIP uses random orthogonal rotations → $\hat{P}$ with no structure
- AWQ uses per-channel scaling → diagonal $\hat{P}$
- **NQP uses a global (or block-wise) $F$ with eigenvalue structure** → a generalization

If NQP = QuIP when $F = I$ (isotropic Fisher), the connection is exact and NQP is strictly
more general.

---

## RQ-6 (Applied)
Can NQP be used as an infrastructure layer for models within CAL?

If NQP produces quantized models with less degradation, the CAL/L2 LLM evaluator (currently
`claude-sonnet-4-6` via API) could eventually be replaced by a local model quantized via NQP
with equivalent quality — reducing the experiment's inference costs and removing the
dependency on an external API.

**Condition:** the NQP-quantized model must pass CAL's φ gate (ρ ≥ 0.75 on the DT-021
benchmark).
