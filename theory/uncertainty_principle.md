# NQP — Weight/Activation Uncertainty Principle

**Status:** Initial draft · 2026-06-24
**Authors:** Juan Pablo Chancay, Claude Sonnet 4.6
**Project:** Natural Quantization via State Preparation (NQP)

---

## 0. Why this pivot

Experiments EXP-001 and A-G4 (see [ROADMAP](../experiments/ROADMAP.md)) showed that the
**quantization** part of NQP has no genuine quantum content: the **activation** Fisher basis
collapses onto the combination GPTQ (second order) + AWQ (outliers) + QuIP (incoherence), and the
analogy with "measuring in the Hamiltonian's basis" turned out to be metaphorical — the empirical
activation Fisher is rank ~2 (PR ≈ 1.5–2.0 out of 768), without the rich spectrum the analogy
requires.

**The genuinely quantum part of NQP is a different one:** the conjecture that there exists an
**uncertainty principle** between weight precision and activation precision. No existing
quantization method addresses this, and here the quantum analogy stops being a metaphor — it
becomes literal mathematical structure (operators that do not commute).

---

## 1. The two Fisher metrics

NQP defines two distinct geometric objects over the same layer:

**Fisher with respect to weights** $F_W$ — how the loss changes when perturbing the **weights**
$W$:
$$F_W = \mathbb{E}_{x}\left[ \nabla_W \mathcal{L}(x)\, \nabla_W \mathcal{L}(x)^T \right]$$
It captures which directions of *weight space* matter for the loss. It is what a weight quantizer
would want to diagonalize.

**Fisher with respect to activations** $G_A$ — how the loss changes when perturbing the
**activations** $a$ (inputs to the layer):
$$G_A = \mathbb{E}_{x}\left[ a\, a^T \right] \quad (\text{Gauss-Newton}) \;\;\text{or}\;\;
\mathbb{E}_{x}\left[ \nabla_a \mathcal{L}\, \nabla_a \mathcal{L}^T \right]$$
It captures which directions of *activation space* matter. It is what GPTQ/AWQ use.

**Key observation:** these live in the same $\mathbb{R}^{d_{in}}$ (via the linear structure
$y = Wa$), so their eigenbases are comparable — and the question of whether they commute is
well defined.

---

## 2. The commutator

Let $\hat{P}_W$ be the basis that diagonalizes $F_W$ and $\hat{P}_A$ the one that diagonalizes
$G_A$.

**Definition 2.1** — *Preparation commutator*
$$[\hat{P}_W, \hat{P}_A] := \hat{P}_W \hat{P}_A - \hat{P}_A \hat{P}_W$$

**Conjecture NQP-U1 (uncertainty principle):**
$$[\hat{P}_W, \hat{P}_A] \neq 0 \;\;\Longrightarrow\;\; \exists\, c > 0 :\;
\varepsilon_W \cdot \varepsilon_A \;\geq\; c$$
where $\varepsilon_W$ is the minimal weight quantization error and $\varepsilon_A$ the activation
one, simultaneously achievable. That is: **you cannot minimize both at once** if the bases do not
commute — exactly like $\Delta x \, \Delta p \geq \hbar/2$.

**Interpretation:** $F_W$ and $G_A$ play the role of two incompatible observables. The basis that
is optimal for quantizing weights is NOT optimal for quantizing activations, and the degree of
incompatibility is quantified by $\|[\hat{P}_W,\hat{P}_A]\|$.

---

## 3. The quantum analog (now literal, not metaphorical)

| Quantum mechanics | NQP-U |
|---|---|
| Position observable $\hat{x}$ | Weight Fisher $F_W$ |
| Momentum observable $\hat{p}$ | Activation Fisher $G_A$ |
| $[\hat{x},\hat{p}] = i\hbar \neq 0$ | $[\hat{P}_W,\hat{P}_A] \neq 0$ |
| $\Delta x \, \Delta p \geq \hbar/2$ | $\varepsilon_W \, \varepsilon_A \geq c$ |
| Incompatible eigenbases | Incompatible quantization bases |
| $\hbar$ (quantum of action) | $c$ (quantum of joint error) |

Unlike quantization (where the analogy was decorative), here the non-commutativity is a
**measurable property** of the network, and the uncertainty bound is **falsifiable**.

---

## 4. Falsifiable predictions

**NQP-U1a:** In real transformer layers, $\|[\hat{P}_W,\hat{P}_A]\| > 0$ in a statistically
significant way (the bases do NOT commute). → *Direct test: measure the commutator.*

**NQP-U1b:** The product $\varepsilon_W \cdot \varepsilon_A$ is bounded below and the bound
correlates with $\|[\hat{P}_W,\hat{P}_A]\|$. → *Test: sweep bit allocations between weights and
activations, trace the Pareto frontier, see if there is a "floor".*

**NQP-U1c (strong):** The bound $c$ predicts the minimal degradation achievable by ANY joint
weight+activation quantization scheme. If true, NQP-U yields a **fundamental limit** — something
neither GPTQ nor AWQ nor QuIP provides (all optimize one side at a time).

---

## 5. Why this IS novel

- GPTQ/AWQ/QuIP quantize **weights** using **activation** geometry. None models the
  weight↔activation tension as a fundamental trade-off.
- Activation quantization (e.g. SmoothQuant) and weight quantization are treated as separate
  problems or balanced heuristically. NQP-U proposes that there is a **theoretical limit** to that
  balance, derived from non-commutativity.
- If NQP-U1c holds, the contribution is not "another quantization method" (a saturated market) but
  an **impossibility bound** — a result of a different kind, closer to information theory than to
  kernel engineering.

---

## 6. Result EXP-U01 (2026-06-24) — NQP-U1a SUPPORTED

Measurement of the eigenbases of $F_W$ (weight Fisher) and $G_A$ (activation Fisher) on GPT-2
small, 64 WikiText-103 calibration samples, dominant top-16 subspace:

| Reference | Principal angle (top-16) | top-eigvec overlap |
|---|---|---|
| Bases commute (same basis) | 0° | 1.000 |
| **GPT-2 observed** | **48.8°** | **0.434** |
| Random bases (control, 5 seeds) | 83.0° ± 0.1 | 0.030 ± 0.02 |

**Reading:** the weight-Fisher and activation-Fisher bases of GPT-2 land at a **genuine
intermediate** point between commuting (0°) and being random (83°):
- **They do not commute** (48.8° >> 0°) → there is a conflict between the basis optimal for
  weights and the one optimal for activations. *The precondition of the uncertainty principle is
  satisfied.*
- **They are not random** (48.8° << 83°; overlap 0.434 vs 0.030 = 14× above chance) → the conflict
  is structural, not trivial. If they were random, any pair of bases would show it and the result
  would mean nothing.
- **It varies by layer type:** `lm_head` gives 21.7° (nearly aligned), attention ~45-48°. The
  structure depends on the layer's function → a real signal, not uniform noise.

**Methodological caveat:** the full-rank commutator norm $\|[\hat{P}_W,\hat{P}_A]\|_F$ **saturates
at 1.0 in high dimension** (any non-identical pair in d=768 saturates it, including random). It is
NOT the decisive metric. The valid metric is the **principal angles of the dominant subspace**
compared against the random control.

## 7. Result EXP-U02 (2026-06-25) — NQP-U1b REFUTED

The $\varepsilon_W$/$\varepsilon_A$ frontier traced over 49 layers (GPT-2, 4-bit, shared
interpolated basis $U(\alpha)$ from $P_A$ to $P_W$, quantizing weights AND activations):

| Metric | Value | Reading |
|---|---|---|
| Interior joint optimum (real trade-off) | 14/49 (29%) | most at an extreme |
| corr($\theta$, log floor) | +0.435 | seemed to support U1b |
| corr($\varepsilon_W$, log floor) | +0.913 | the floor ≈ $\varepsilon_W$ |
| **partial corr($\theta$, floor \| $\varepsilon_W$)** | **−0.041** | **the angle adds nothing** |

**The angle↔floor correlation was spurious.** Controlling for $\varepsilon_W$ (which dominates the
floor, corr 0.91), the angle's effect vanishes (partial −0.04 ≈ 0). The angle correlates with the
floor only because *both* track the intrinsic difficulty of quantizing weights
($\theta$↔$\varepsilon_W$ = 0.49), not because of an uncertainty relation.

**Conclusion:** there is NO bound $\varepsilon_W\cdot\varepsilon_A \geq c(\theta)$. The
non-commutativity of the bases (U1a, real) **does not translate** into a quantization trade-off
governed by the angle. A scale asymmetry ($\varepsilon_W \gg \varepsilon_A$ at equal bits)
dominates and masks any geometric effect.

## 8. Status of the uncertainty conjectures

- **NQP-U1a** (bases do not commute): ✅ supported — but it is a *static* geometric property, it
  does not imply an operational consequence.
- **NQP-U1b** (bound $\varepsilon_W\varepsilon_A \geq c(\theta)$): ❌ refuted — the angle does not
  predict the floor once controlling for $\varepsilon_W$.
- **NQP-U1c** (fundamental limit of joint quantization): ❌ falls with U1b.

**Methodological lesson:** the bivariate correlation (+0.44) looked like evidence; the *partial*
correlation (−0.04) dismantled it. Always control for the obvious confound ($\varepsilon_W$) before
attributing a correlation to the pretty hypothesis.
