# NQP — Roadmap of checks A → B → C

**Last updated:** 2026-06-24
**Starting point:** EXP-001 (diagonal Fisher, $\hat{P}=I$) implemented and validated.

**Result of the calibration sweep (2026-06-24):**

| n_calib | bits | PPL std | PPL NQP | Δ | NQP wins (L2 err) |
|---|---|---|---|---|---|
| 32 | 8 | 34.06 | 36.67 | +2.61 | 0/48 |
| 64 | 8 | 34.06 | 36.62 | +2.55 | 0/48 |
| 128 | 8 | 34.06 | 37.02 | +2.96 | 0/48 |
| 128 | 4 | 9808 | 7175 | (both garbage) | **20/48** |

**Reading:**
- **8 bits → flat, dead diagonal.** Δ does not move with calibration (32→128). The problem
  is structural ($\hat{P}=I$), not a matter of samples. **Case closed.**
- **4 bits → a raw lever appears.** 20/48 layers reduce L2 error (vs 0/48 at 8 bits), but
  both PPLs are garbage (4-bit RTN destroys GPT-2) and the diagonal cannot convert the
  L2-error reduction into PPL because it lacks the rotation.
- **A clear signature that $\hat{P}\neq I$ is missing.** The 4-bit signal is exactly what
  Path A should be able to exploit — and the correct 4-bit baseline is GPTQ, not RTN.

**Entry gate to Path A: PASSED.**

---

## Why A → B → C (and not some other order)

NQP's central intuition (CLAUDE.md §"Foundational intuition") is to *quantize in the
**eigenbasis** of the model's natural operator* — analogous to measuring in the Hamiltonian's
basis. That eigenbasis is $\hat{P}=U$ with $U^T F U = \Lambda$
([operator_formalization.md:58](../theory/operator_formalization.md)).

**The diagonal tested in EXP-001 has $U=I$: no eigenbasis, no rotation.**
It reduces to adaptive per-element scaling (≈ AWQ). That is why it loses. The NQP-C1 hypothesis
**was never really tested** — we tested its trivial shadow.

The three paths are **increasing levels of fidelity to the intuition**, ordered by
signal/effort ratio:

| | Path | $\hat{P}$ | Embodies the intuition | Effort | Role in the roadmap |
|---|---|---|---|---|---|
| **A** | Block-wise Fisher + real rotation | $U \neq I$ (block eig) | ✅ Literal | Medium | **Existence proof** of NQP-C1 |
| **B** | K-FAC (Kronecker-factored) | $U_A \otimes U_G$ | ✅ Scalable | High | **Scales** A to real LLMs |
| **C** | Rate-distortion over the eigenspectrum | $U$ + optimal bit allocation | ✅✅ Strong form | High | **Theorem** (formal NQP-C1/C2) |

A is the lowest-risk bet that touches the real idea. B only makes sense if A works.
C is the paper, and is only written if A+B give empirical signal.

---

## PATH A — Block-wise Fisher with real rotation

**Objective:** instantiate $\hat{P}=U \neq I$ and check whether quantizing in the Fisher basis
(not random, unlike QuIP) beats the baseline at low bits.

**Operational definition:**
- For each weight matrix $W \in \mathbb{R}^{d_{out}\times d_{in}}$, estimate the block Fisher
  over the input columns: $F \in \mathbb{R}^{d_{in}\times d_{in}}$ (Gauss-Newton / empirical
  Fisher of input activations — same structure as GPTQ's Hessian).
- Diagonalize: $F = U\Lambda U^T$.
- Rotate the weights into the eigenbasis: $\tilde{W} = W U$.
- Quantize $\tilde{W}$ with a per-column scale derived from $\lambda_i$.
- Reconstruct: $\hat{W} = Q(\tilde{W})\,U^T$.

**Checks (gates):**
- **A-G1** — sanity: $U^TU = I$ (numerical orthogonality), FP32 reconstruction without
  quantizing recovers $W$ with error < 1e-5.
- **A-G2** — L2 error: $\varepsilon_{NQP} < \varepsilon_{std}$ in ≥ 60% of layers at **4 bits**
  (at 8 bits there is little signal; the interesting regime is 4/3 bits).
- **A-G3** — PPL: $\text{PPL}_{NQP} \leq \text{PPL}_{GPTQ}$ on GPT-2 at 4 bits (GPTQ is the
  fair comparator, not INT4-RTN).
- **A-G4** — key ablation: does the **Fisher** rotation beat a **random** rotation
  (QuIP) at the same budget? If NOT, the Fisher structure adds nothing and NQP collapses
  onto QuIP. **This is the gate that decides whether NQP exists as a method.**

**Outcome:** if A-G4 passes → NQP has empirical content → proceed to B.
If A-G4 fails → the intuition is elegant but the Fisher basis does not beat chance; pivot
to investigating *why* (poorly estimated Fisher? block too small?) before abandoning.

### A-G4 v1 result (2026-06-24) — FAIL, but diagnostic

| basis | mean L2 err (4-bit) |
|---|---|
| RTN (no rotation, U=I) | **4.97e-2** (best) |
| Fisher (U≠I) | 5.10e-2 |
| Random (QuIP) | 5.27e-2 (worst) |

- Fisher beats random in only 16/49 layers (33%) → **FAIL of the gate as defined**.
- **BUT**: on average, *rotating makes it worse* (both > RTN), and Fisher < random. The gate
  assumed rotation helps (the QuIP model); here, rotating with a **per-`max|col|` quantizer
  and uniform bits** hurts because it disperses weights that were concentrated.
- **Root cause (not a bug, an incomplete theory):** `quantize_rotated` implemented the rotation
  but NOT the **adaptive per-eigenvalue bit allocation** that formalism §4 marks as the active
  ingredient ($b_i \propto \lambda_i$, scale $\propto \lambda_i^{-1/2}$). We rotated into the
  Fisher basis and then ignored Fisher when quantizing. That wastes exactly what Fisher
  optimizes.

**Verdict:** A-G4 v1 does not falsify NQP — it falsifies "Fisher rotation + naive quantizer".
The gate must be re-run with **scale derived from λ_i** (A-G4 v2) before concluding anything
about the central hypothesis.

### A-G4 v2 result (2026-06-24) — strong FAIL, and reveals the measurement error

| basis | mean L2 err (4-bit) | wins in |
|---|---|---|
| RTN (U=I) | 4.97e-2 | 7/49 |
| Fisher-naive (U≠I, max-col) | 5.10e-2 | 11/49 |
| Random (QuIP) | 5.27e-2 | 31/49 |
| **NQP\*** (U≠I, λ scale) | **9.08e-2** | **0/49** (worst of all) |

The λ-scale **doubled** the error instead of lowering it. Definitive diagnosis:

> **Gate A-G4 measures L2 reconstruction error, but NQP does NOT promise to minimize L2 —
> it promises to minimize the impact on the LOSS (perplexity).** By design, the λ-scale
> *sacrifices* L2 to protect loss-sensitive directions. Measuring L2 is structurally biased
> against NQP's idea: RTN always wins because RTN minimizes exactly what the gate measures.
> It is a tautology, not evidence.

**Conclusion of the L2-error phase:** no L2-error-based experiment can validate NQP. Every
comparison must be in **PPL / downstream loss**, where protecting high-curvature directions
can pay off. L2 error only serves as a sanity check that the reconstruction is not broken —
not as a metric of merit.

### Re-plan of Path A (A v3)
- Merit metric: **ΔPPL**, not L2 error.
- Honest comparator: **GPTQ** (which also optimizes loss impact via the Hessian), not RTN.
- Refined hypothesis: Fisher rotation + protection of curved directions should beat GPTQ
  *in PPL at 3-4 bits*, even if it loses in L2. If it does not win in PPL either, then the
  Fisher basis of input activations adds nothing over the output Hessian GPTQ already uses,
  and NQP's value (if any) lies in C (rate-distortion over the eigenspectrum), not in A.

**Estimated cost:** ~3–4 sessions. Diagonalizing 768×768 blocks is trivial on CPU.

---

## PATH B — K-FAC (Kronecker-factored)

**Precondition:** A-G4 passed (the Fisher basis adds over chance).

**Objective:** make A scalable. The full block Fisher is $O(d^2)$ per layer; infeasible at
7B+. K-FAC factors $F \approx A \otimes G$ (input-cov ⊗ output-grad-cov), reducing it to two
small eigendecompositions and $\hat{P} = U_A \otimes U_G$.

**Checks (gates):**
- **B-G1** — fidelity: the K-FAC basis reproduces ≥ 90% of A-G3's gain on GPT-2 (verify that
  the Kronecker approximation does not destroy the signal).
- **B-G2** — scale: run on Llama-3 8B at 4 bits, $\text{PPL}_{NQP} \leq \text{PPL}_{GPTQ}$
  (= EXP-002 of the README).
- **B-G3** — overhead: time to prepare $\hat{P}$ < 2× that of GPTQ (RQ-3: overhead vs quality
  trade-off).

**Outcome:** if B passes → NQP is a viable, competitive quantization method → C.

**Estimated cost:** ~6–8 sessions (K-FAC implementation + GPU access for Llama-8B).

---

## PATH C — Rate-distortion over the eigenspectrum

**Precondition:** A+B give reproducible empirical signal.

**Objective:** the strong form. Not just rotate, but **allocate bits optimally** per
eigenvalue via rate-distortion: $b_i \propto \log \lambda_i$ (more bits where the loss is
more curved), and formalize NQP-C1 as a theorem with sufficient conditions
([operator_formalization.md:104](../theory/operator_formalization.md)).

**Checks (gates):**
- **C-G1** — theory: derive the optimal bit allocation under a fixed budget and prove the
  bound $\varepsilon_{NQP} \leq \varepsilon_{std}$ under exact Fisher.
- **C-G2** — NQP-C2 (strong form): empirical test of whether NQP-4bit **beats FP32** on tasks
  where the calibration is representative (= EXP-003, the regularizer hypothesis).
- **C-G3** — paper: reproducible results + comparison vs GPTQ/QuIP#/AWQ.

**Estimated cost:** open-ended — it is the body of the paper.

---

## Decision on "which path to exploit"

**Design verdict:** exploit **A first, in depth**, because:

1. It is the **only** one that can *falsify* NQP cheaply. A-G4 (Fisher vs random rotation) is
   the central scientific question — if the Fisher basis does not beat chance, everything else
   is decoration. No other path answers this faster.
2. B and C **inherit** all their validity from A. Investing in K-FAC or in rate-distortion
   theory before knowing whether the Fisher rotation adds anything is building on sand.
3. A reuses ~70% of the current code (`fisher.py`): it only adds the diagonalization +
   rotation block. The diagonal already built remains as an ablation ($U=I$).

**Anti-pattern to avoid:** jumping to B (K-FAC, "the scalable one") because it sounds more
impressive. If A does not pass A-G4 on GPT-2 (124M, minutes of compute), B on Llama-8B (hours
of GPU) will only waste resources confirming the same thing at higher cost.

---

## Next concrete action

Implement `src/fisher_block.py` (Path A) with:
- `estimate_block_fisher(model, calib)` → $F$ per weight matrix.
- `NQPBlockQuantizer` with rotation $\tilde W = WU$ / reconstruction $\hat W = Q(\tilde W)U^T$.
- A-G4 comparator: NQP-Fisher vs NQP-random-rotation vs GPTQ-baseline.
- Entry gate: wait for the calibration-sweep verdict (EXP-001) to confirm that the diagonal is
  flat before investing in A.
