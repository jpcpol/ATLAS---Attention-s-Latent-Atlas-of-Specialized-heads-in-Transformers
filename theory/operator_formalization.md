# NQP — Formalization of the State Preparation Operator

**Status:** Initial draft · 2026-06-24
**Authors:** Juan Pablo Chancay, Claude Sonnet 4.6
**Project:** Natural Quantization via State Preparation (NQP)

---

## 1. Motivation

Standard LLM quantization imposes a uniform discrete grid on a continuous weight space. This
introduces quantization noise because the grid does not respect the real geometry of the weight
distribution: some critical weights collapse to incorrect bins, while high-resolution bins are
wasted in low-variance regions.

**Central intuition (J.P. Chancay, 2026-06-24):** just as in quantum mechanics a system in
superposition collapses to an eigenvalue *of the measurement operator*, a model's weights should
collapse to discrete bins *of the model's natural operator* — not of an external grid. Optimal
quantization is the one in which the grid emerges from the model's geometry rather than being
imposed on it.

---

## 2. Base definitions

Let $W \in \mathbb{R}^{d}$ be the weight vector of a layer (or group of layers) of an LLM.

**Standard quantization** applies:

$$Q_{\text{std}}(W) = s \cdot \text{round}\!\left(\frac{W}{s}\right) + z$$

with $s$ (scale) and $z$ (zero-point) fixed by the statistical range of $W$. The error is:

$$\varepsilon_{\text{std}} = \|W - Q_{\text{std}}(W)\|^2$$

**NQP goal:** find a transformation $T: \mathbb{R}^d \to \mathbb{R}^d$ such that:

$$\varepsilon_{\text{NQP}} = \|W - T^{-1}(Q_{\text{std}}(T(W)))\|^2 \ll \varepsilon_{\text{std}}$$

with $T$ invertible, efficiently computable, and preserving the model's inferential capacity.

---

## 3. The preparation operator $\hat{P}$

**Definition 3.1** — *Natural preparation operator*

Let $\mathcal{H}_W$ be the weight space with an inner product defined by the model's Fisher
metric:

$$\langle u, v \rangle_F = \mathbb{E}_{x \sim \mathcal{D}}\left[u^T \nabla^2_W \mathcal{L}(x; W) \, v\right]$$

where $\mathcal{D}$ is the calibration distribution and $\mathcal{L}$ is the model's loss.

The preparation operator $\hat{P}$ is the orthogonal (or quasi-orthogonal) linear transformation
that diagonalizes the local Fisher metric:

$$\hat{P} = U \quad \text{such that} \quad U^T F U = \Lambda$$

where $F$ is the Fisher matrix (or its block approximation) and $\Lambda$ is diagonal.

**Interpretation:** in the basis of $\hat{P}$, the weight-space directions are *independent under
the loss* — analogous to the eigenvector basis of the Hamiltonian in QM, where each direction has
a defined energy (impact).

---

## 4. Quantization in the natural basis

After applying $\hat{P}$:

$$\tilde{W} = \hat{P} W$$

each component $\tilde{W}_i$ has variance proportional to $\lambda_i^{-1}$ (the inverse of the
corresponding Fisher eigenvalue). This enables:

1. **Adaptive bit allocation:** components with large $\lambda_i$ (high loss curvature → high
   sensitivity) receive more bits; components with small $\lambda_i$ receive fewer.

2. **Non-uniform grid:** the discrete bins are distributed according to the variance of
   $\tilde{W}_i$, not uniformly.

3. **Quantization:** $\hat{W}_i = Q_i(\tilde{W}_i)$ with $Q_i$ specific to each component.

4. **Reconstruction:** $\hat{W} = \hat{P}^{-1} \hat{\tilde{W}}$

---

## 5. Explicit quantum analog

| Quantum mechanics | NQP |
|---|---|
| Pre-measurement state $\|\psi\rangle$ | FP32 weights $W$ |
| Superposition in the computational basis | Continuous distribution in $\mathbb{R}^d$ |
| Hamiltonian $\hat{H}$ | Fisher metric $F$ |
| Change to the eigenvector basis of $\hat{H}$ | Transformation $\hat{P} = U$ (diagonalization of $F$) |
| Energy eigenvalues $E_n$ | Fisher eigenvalues $\lambda_i$ (loss curvature) |
| Collapse to the nearest eigenvalue | Quantization $Q_i$ in the natural basis |
| "EM field" that prepares the state | Fine-tuning / calibration that nudges $W$ toward the bins |
| Minimal measurement error in the eigenbasis | Minimal quantization error in the Fisher basis |

---

## 6. Target property (conjecture NQP-C1)

**Conjecture:** for any model $M$ with weights $W$ and any bit budget $b$, there exists a
preparation operator $\hat{P}$ such that:

$$\text{PPL}(M_{\hat{P},b}) \leq \text{PPL}(M_{\text{std},b}) + \delta$$

with $\delta \to 0$ as the number of calibration samples $n \to \infty$, where $\text{PPL}$ is the
perplexity of the quantized model over an evaluation distribution and $M_{\hat{P},b}$ is the model
quantized via NQP with $b$ bits.

**Stronger form (NQP-C2):** the optimal $\hat{P}$ makes $\delta < 0$ — that is, quantization in
the natural basis with $b$ bits beats unquantized FP32 on tasks where the calibration distribution
is representative, because $\hat{P}$ acts as a natural regularizer.

---

## 7. Connection to existing work

| Method | Relation to NQP |
|---|---|
| GPTQ | Minimizes per-layer quantization error using the output Hessian — a block approximation of $F$ |
| QuIP / QuIP# | Applies a random orthogonal rotation — a special case of $\hat{P}$ with no Fisher structure |
| AWQ | Weights the error by activations — a diagonal approximation of $\hat{P}$ |
| SmoothQuant | Per-channel rebalancing — a 1D case of the transformation |
| **NQP** | Generalization: optimal $\hat{P}$ given $F$, with adaptive per-eigenvalue bit allocation |

NQP's novelty over QuIP is that the rotation **is not random** — it is the model's Fisher basis,
which has meaning in terms of loss sensitivity.

---

## 8. Open questions

1. Is $F$ efficiently computable for LLM-scale models (7B–70B)?
   → Approximations: K-FAC, diagonal, per-layer block.

2. Does conjecture NQP-C2 (strong form) hold in practice?
   → Experiment: compare the PPL of NQP vs GPTQ vs QuIP on Llama-3 8B at 4 bits.

3. What is the complexity of computing $\hat{P}$ vs the saving in inference quality?
   → Preparation-overhead vs quality-gain trade-off.

4. Is there a notion of an "uncertainty principle" in NQP?
   → If $\hat{P}$ diagonalizes $F$, are there directions where weight precision and activation
   precision cannot be optimized simultaneously?

---

## 9. Next steps

- [ ] Implement diagonal Fisher estimation for a small transformer (GPT-2)
- [ ] Compare $\varepsilon_{\text{NQP}}$ vs $\varepsilon_{\text{std}}$ on a controlled distribution
- [ ] Verify conjecture NQP-C1 empirically
- [ ] Formalize NQP-C2 as a theorem with sufficient conditions
