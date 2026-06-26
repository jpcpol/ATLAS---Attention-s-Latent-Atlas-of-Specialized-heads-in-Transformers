# Mathematical Map: Quantum Mechanics ↔ Transformers

**Status:** Initial exploration · 2026-06-25
**Authors:** Juan Pablo Chancay, Claude Opus 4.8
**Origin:** successor line to NQP (on standby). NQP failed by starting from ONE forced analogy
(quantization = measurement). Here we invert the approach: systematically map the shared
mathematical machinery, identify gaps where QM has structure that transformers do not exploit,
and filter by **falsifiable viability**, not by elegance.

---

## 0. Anti-NQP discipline

Each candidate must pass three filters before investing compute:

1. **Literal, not metaphorical, match:** do QM and transformers use *the same mathematical
   object*, or do they only "look alike"? (NQP confused "Fisher looks like the Hamiltonian" with
   "Fisher is the Hamiltonian". It failed.)
2. **Real gap:** does the QM tool do something the transformer's current formulation does NOT
   already do under another name? (NQP-C1 collapsed onto GPTQ+AWQ+QuIP because it added nothing
   new.)
3. **Falsifiable and measurable:** is there a cheap experiment that can *refute* the usefulness on
   GPT-2 before scaling? (NQP-U1b looked supported until the partial correlation.)

---

## 1. LITERAL mathematical matches

Objects both fields use identically (not analogously):

| Object | In QM | In transformers | Literal? |
|---|---|---|---|
| **Inner product / projection** | ⟨ψ\|φ⟩ amplitude | QᵀK attention score | 🟢 same algebra |
| **Softmax / Gibbs distribution** | e^{−E/kT}/Z (Boltzmann) | softmax(QKᵀ/√d) | 🟢 same form, T↔√d |
| **L2 normalization / state** | \|ψ\|=1 (normalized state) | LayerNorm, RMSNorm | 🟡 partial (L2 vs variance) |
| **Linear operator / dense matrix** | Â Hermitian observable | W proj (non-Hermitian) | 🟡 form yes, Hermiticity no |
| **Superposition / linear combination** | Σ cᵢ\|i⟩ | Σ attn_i · vᵢ (mixture of values) | 🟢 same convex structure |
| **Tensor product / composition** | H_A ⊗ H_B composite systems | multi-head (concat spaces) | 🟡 concat ≠ full ⊗ |
| **Density matrix ρ** | ρ=Σ pᵢ\|ψᵢ⟩⟨ψᵢ\| mixed state | — (not used explicitly) | ⬜ candidate GAP |

**Key observation:** attention IS, formally, an expected value under a Boltzmann distribution over
states (the values), with the energy given by −QᵀK/√d. This is NOT a metaphor: it is the same
formula. It is the strongest literal anchor and the natural starting point.

---

## 2. GAPS — structure QM has that transformers do NOT exploit

Ordered by estimated viability (filter §0), not by conceptual appeal.

### GAP-A — Density matrix / mixed states (attention as ρ)
Attention produces a convex mixture of values: `out = Σ_i a_i v_i`. In QM that is exactly a
**mixed state** described by a density matrix ρ. Transformers throw away that structure: they
collapse the mixture to its mean (the output vector) and lose the *coherence* (the cross terms
⟨v_i\|v_j⟩).
- **Gap:** the attention output retains only `Tr(ρ V)` (the mean) and discards the whole matrix
  ρ = Σ a_i \|v_i⟩⟨v_i\|, which encodes the *uncertainty/spread* of the mixture.
- **Useful hypothesis:** propagating a second-order statistic (variance of the mixture, or the
  "purity" Tr(ρ²)) could give upper layers a signal of *how confident/dispersed* each attention
  decision was — cheap and possibly useful for calibration/uncertainty.
- **Cheaply falsifiable:** yes. Compute Tr(ρ²) per head in GPT-2 and see if it correlates with
  anything (prediction entropy, errors).

### GAP-B — Unitary evolution / reversibility
In QM, evolution without measurement is **unitary** (preserves norm and information; reversible).
Transformer blocks are NOT unitary (LayerNorm, ReLU/GeLU, residuals make them
contractive/expansive and irreversible).
- **Gap:** networks with unitary/orthogonal layers have gradients that neither explode nor vanish
  (norm preserved) — relevant for stability at extreme depth.
- **State of the art:** prior work exists (unitary RNNs, orthogonal init, Stiefel
  parametrizations). It is NOT virgin territory. Risk of collapsing onto existing literature (like
  NQP→GPTQ).
- **Falsifiable:** yes, but the real gap over existing work is doubtful. Medium-low priority.

### GAP-C — Entanglement / non-factorizability as a metric
**Entanglement** measures how much a composite state does NOT factor: whether
\|ψ_AB⟩ ≠ \|ψ_A⟩⊗\|ψ_B⟩. The entanglement entropy (von Neumann of the reduced ρ) quantifies
"genuinely joint" correlation.
- **Gap:** there is no standard metric in transformers of how much one token's representation is
  "entangled" with another's (beyond the raw attention weight). The entropy of the reduced density
  matrix would give a principled measure.
- **Useful hypothesis:** it could be an *interpretability* tool (which tokens form inseparable
  semantic units) more than a performance one.
- **Cheaply falsifiable:** yes, measurable on GPT-2 activations.

### GAP-D — Phase formalism / complex amplitudes
QM lives in ℂ: amplitudes have a **phase**, and interference (constructive/destructive) is the
central mechanism. Transformers are purely real — there is no interference.
- **Gap:** attention only *adds* (always constructive interference). It cannot *cancel*
  contributions via opposite phase. A complex-valued attention mechanism could natively represent
  "this token contradicts that one".
- **State of the art:** complex/phase-aware transformers exist, mixed results.
- **Falsifiable:** medium (requires modifying the architecture and retraining — expensive in this
  setup).

### GAP-E — Variational principle / free energy
QM and statistical mechanics minimize **free energy** F = E − TS (energy/entropy balance).
Inference in transformers has no explicit energy functional minimized in the forward pass.
- **Gap/connection:** modern Hopfield networks (Ramsauer et al.) already showed that attention IS
  the update of a Hopfield-type energy model. This links attention to an energy landscape — and
  QM/statistics have rich tools there (temperature, phase transitions, annealing).
- **Useful hypothesis:** does dynamic "temperature" control in attention (annealing) improve
  multi-step reasoning? Connects with the √d as inverse temperature (§1).
- **Cheaply falsifiable:** YES — it only modifies the attention scaling at inference, without
  retraining. High value/cost ratio.

---

### GAP-F — Helmholtz free energy (J.P. Chancay, 2026-06-25)
With $A_i = e^{\beta s_i}/Z$, $s_i = q\cdot k_i/\sqrt{d}$, $\beta = $ inverse temperature, the
quantity $F = -\tfrac{1}{\beta}\log Z$ **is** the Helmholtz free energy. Attention computes $Z$
but never uses $F$ or its derivatives.
- **Gap:** in statistical physics, phase transitions are detected in the *derivatives* of $F$
  (heat capacity, susceptibility). Transformers ignore all that structure.
- **Hypothesis:** different head types (syntactic / fact-retrieval / reasoning) could operate in
  different thermodynamic regimes, some near a critical point.
- **Cheaply falsifiable:** yes — vary $\beta$ at inference, measure PPL/entropy/stability, look for
  peaks in $C = \partial\langle E\rangle/\partial T$ and $\chi = \partial\langle A\rangle/\partial T$.

### GAP-G — Mutual information between heads (J.P. Chancay)
Attention entropy and rollout are measured, but almost no one measures $I(H_i; H_j)$ between
heads. In statistical mechanics, **long-range correlations = an indicator of criticality**.
- **Hypothesis:** important heads could exhibit high correlation / synchronization on certain
  tokens. This would allow pruning and compression WITHOUT retraining.
- **Cheaply falsifiable:** yes (only measuring activations).

### GAP-H — von Neumann entropy (extension of GAP-A, J.P. Chancay)
Purity $\mathrm{Tr}(\rho^2)$ is just a 2nd-order moment. The von Neumann entropy
$S(\rho) = -\mathrm{Tr}(\rho\log\rho)$ contains much more information.
- **Hypothesis:** when hallucinating / being uncertain / facing OOD, $S(\rho)$ rises and purity
  drops → a **practically free uncertainty estimator**.
- **Cheaply falsifiable:** yes.

### GAP-I — Renormalization Group (RG) (J.P. Chancay) — high potential, more theoretical
A transformer is a sequence of mixtures/projections/compressions $x_l \to x_{l+1}$ that removes
irrelevant degrees of freedom — exactly Wilson's RG **coarse-graining**.
- **Questions:** do the layers implement coarse-graining? are there fixed points? universality
  classes between models? It could explain scaling laws and optimal depth.
- **Falsifiable:** medium-expensive (requires multi-layer/multi-model analysis), Phase 3.

### GAP-J — Information geometry (J.P. Chancay)
The attention distribution is a statistical distribution → it has a Fisher metric, curvature,
geodesics.
- **Hypothesis:** errors concentrate in high-curvature regions → adaptive attention / local
  temperatures / routing, without touching training. (Caution: NQP already tripped on Fisher;
  here it is the Fisher of the *attention distribution*, a different and better-defined object.)

### GAP-K — Heat capacity as a detector of cognitive regime (J.P. Chancay)
$C = \beta^2(\langle E^2\rangle - \langle E\rangle^2)$. In physics, peaks of $C$ ⇒ a phase
transition. In LLMs they could mark regime changes: memory retrieval vs multi-hop reasoning vs
creative generation. **Very little explored.**
- **Cheaply falsifiable:** yes — $C$ is the variance of the energy, computable from the attention
  logits that already exist.

---

## 3. Viability ranking (candidates to explore)

Unifying insight (J.P. Chancay): **all the thermodynamics derive from the same $Z$ that attention
already computes.** $F$, $\langle E\rangle$, $C$, $\chi$, $S(\rho)$ are functions of the same
attention logits — measuring them is nearly free and requires no retraining. This raises the whole
thermodynamic block (E/F/H/K) to joint HIGH priority.

| Gap | Literal match | Real gap vs SOTA | Cheaply falsifiable | Phase |
|---|---|---|---|---|
| **GAP-E** (temperature) | 🟢 softmax=Boltzmann | 🟡 Hopfield connects | 🟢 no retraining | **1** |
| **GAP-A** (purity ρ) | 🟢 mixture=ρ | 🟢 not used | 🟢 measure only | **1** |
| **GAP-H** (von Neumann S) | 🟢 mixture=ρ | 🟢 not used | 🟢 measure only | **1** |
| **GAP-K** (heat capacity) | 🟢 C=var(E) | 🟢 unexplored | 🟢 measure only | **1** |
| **GAP-F** (free energy) | 🟢 F=−logZ/β | 🟢 derivatives unused | 🟢 measure only | 2 |
| **GAP-G** (head mutual info) | 🟡 medium | 🟢 rarely measured | 🟢 measure only | 2 |
| **GAP-I** (renormalization) | 🟡 coarse-grain | 🟢 explains scaling | 🔴 multi-layer/model | 3 |
| **GAP-J** (info geometry) | 🟡 medium | 🟡 (NQP tripped) | 🟡 | 3 |
| GAP-C (entanglement) | 🟡 | 🟢 | 🟢 | 2 |
| GAP-D (complex phase) | 🟡 | 🟡 exists | 🔴 retrain | low |
| GAP-B (unitarity) | 🟡 | 🔴 much SOTA | 🟡 | low |

---

## 4. Experiment plan

### Phase 1 (no retraining) — thermodynamic characterization
**EXP-Q01 — Thermodynamic Characterization of Transformer Attention.** Per attention head in
GPT-2, measure from the attention logits $s_i = q\cdot k_i/\sqrt d$:
- effective temperature (softmax dispersion) vs $\sqrt d$  [E]
- free energy $F = -\tfrac1\beta\log Z$  [F]
- purity $\mathrm{Tr}(\rho^2)$  [A]
- von Neumann entropy $S(\rho)$  [H]
- heat capacity $C = \beta^2(\langle E^2\rangle-\langle E\rangle^2)$  [K]
- **correlate all with:** perplexity, output entropy, (later) hallucination, OOD.

Central question: are there head types with distinct thermodynamic signatures? Does any operate
near a critical point (peaks of $C$/$\chi$)?

### Phase 2 (structural analysis)
- **EXP-Q02:** mutual information $I(H_i;H_j)$ between heads; long-range correlations as a
  criticality detector → pruning/compression without retraining  [G]
- $\beta$ sweep at inference: look for peaks in $C$/$\chi$ = internal phase transitions [F,K]
- entanglement between token representations [C]

### Phase 3 (more theoretical)
- **EXP-Q03:** do the layers implement RG-type coarse-graining? fixed points? universality?
  Connection with scaling laws and optimal depth  [I]
- information geometry of the attention distribution  [J]

**§0 discipline in each phase:** measure before modifying, refute cheaply before scaling, and
always control for the obvious confound (lesson from NQP-U1b: partial, not bivariate, correlation).

### Incidental finding (EXP-Q01 debug, 2026-06-25): a temperature gradient with depth
GPT-2's attention logits grow brutally with depth (L0 max≈38, L6≈4.9e5, L11≈6.2e5 — the known
*attention logit growth* / massive activations phenomenon). In thermodynamic language: **the layers
have an effective temperature gradient.** Early = hot (distributed attention, mixing); deep = cold
(pure state, purity≈1, T_eff≈1, nearly deterministic attention). It emerges without anyone training
it that way. It refines GAP-F/K: the regime change is not only between *head types* but *along the
depth* — final layers in a pure state ≈ "deterministic decision", early ones ≈ "exploration".
Practical implication: any thermodynamic metric must be normalized per layer, or depth dominates it
(a confound to control).

### Result EXP-Q01 (2026-06-25)
- **Phase structure empirically confirmed.** The thermodynamic regime varies strongly:
  - Early layers (L0): "liquid" heads (T_eff≈15, purity≈0.43, S_vn≈1.4) AND "frozen" ones
    (T_eff≈1.2, purity≈0.99) coexist in the same layer.
  - Deep layers (L11): ALL crystallized (T_eff=1.0, purity=1.0, S_vn=0) — deterministic attention,
    collapsed to 1 token.
- **S_vn tracks prediction uncertainty:** corr(S_vn, pred entropy) = −0.20, and **survives the
  control for position** (partial −0.18). Not a confound (in contrast to NQP-U1b).
  → a cheap uncertainty estimator, a real candidate (GAP-H).

### Crystallization hypothesis (J.P. Chancay, 2026-06-25) — NEW PRIORITY
If a head has T_eff→1 (S_vn→0), its softmax collapses to δ and attention stops being a weighted
average: `Attn ≈ V_{argmax}`. In that regime exp/sum/division are **unnecessary** — replaced by
`argmax` (Top-1) or Top-2. Benefit: less compute, less memory (no need to store the attention
matrix), less memory traffic (the real bottleneck of LLMs).

**Strong form — total crystallization:** deep layers might not need attention; approximable by
h_{l+1}=f(h_l) with f = MLP / routing / piecewise linear. Deep attention would just be the
mechanism that *implements* an already-crystallized discrete decision.

**Trap to watch:** S_vn≈0 does NOT imply replaceable — ∂Attn/∂q may still matter if the model
operates *near a transition* (a sensitive region). That is why one must measure and ablate, not
assume. Genuine-determinism metric: R = p₂/p₁ (ratio of the 2nd to the 1st weight). R≪1 ⇒
genuinely deterministic.

### Target paper: *"Thermodynamic Phase Transitions and Attention Crystallization in
### Transformer Networks"*
No longer an analogy: it is a concrete compression/acceleration mechanism derived from the observed
thermodynamics. Central result = the ΔPPL(L) curve of progressive replacement by Top-k.

### EXP-Q02 plan — Attention Crystallization

**Pre-registration (J.P. Chancay, 2026-06-25) — fix the hypotheses BEFORE seeing the data
(anti-NQP):**
- **H0 (null):** the low entropy of deep layers does NOT allow deterministic replacement without
  significant degradation.
- **H1:** there exists a subset of deep layers whose attention is well-approximated by a Top-1/Top-k
  operator with negligible performance loss and an efficiency gain.

**Mandatory controls (low S_vn alone is NOT enough):**
- **C1 — margin/ratio:** R = p₂/p₁ and margin p₁−p₂. Low S_vn can come from [0.51,0.49,…]
  (not deterministic). Genuine crystallization requires R≪1.
- **C2 — noise stability:** P(argmax(q) = argmax(q+ε)). If the argmax changes under a small
  perturbation, the head is *near a transition* → replacing it breaks the model (the "trap":
  deterministic ≠ replaceable).

**Phases:**
- **A (diagnostic, implemented):** per head/layer, measure R and argmax-stability (C1+C2).
- **B (causal):** progressive replacement L11→Top-k, L10-11, … The ΔPPL(L) curve reveals the
  freezing point L_c. Ideal result: ΔPPL≈0 up to some depth, then it explodes.
- **C (strong form):** are crystallized layers replaceable by f(h) without Q/K/attention? →
  hybrid transformers: early thermodynamic layers + final deterministic ones.
- **Metrics:** PPL, exact match, throughput, VRAM, latency.

If B comes out positive even just in the last 2 layers: it stops being characterization and becomes
**an acceleration/compression mechanism derived from a measured thermodynamic property** — a much
stronger publishable result.

### Result EXP-Q02 (2026-06-25) — H1 REFUTED, but with findings

**Phase A (diagnostic):** L1-L11 pass C1+C2 (R≈0.00-0.05, argmax stable >94%). L0 is the only
"liquid" one (R=0.40). The diagnostic said: 11 crystallizable layers.

**Phase B (causal):** Top-k replacement REFUTES the diagnostic:

| replacement | Top-1 ΔPPL | Top-2 ΔPPL |
|---|---|---|
| L11 only | +7.68 | +2.54 |
| L9-11 (3 layers) | +16.5 | +4.71 |
| L3-11 (9 layers) | +65.4 | +12.5 |
| L1-11 | +523 | +78 |
| L0-11 (all) | +2308 | +500 |

**Conclusion: H0 holds, H1 fails.** Replacing even L11 ALONE (R=0.00, argmax 100% stable) by
argmax raises PPL +7.7 — not negligible.

**The lesson — C2 (argmax stability) is necessary but NOT sufficient:**
> Even though p₂/p₁≈0, attention sums over hundreds of tokens with tiny weights. That **tail adds
> a systematic contribution** to the output value. The argmax captures *where* the head looks but
> discards the *integration over context*. S_vn≈0 measures concentration of *where*, not
> irrelevance of the *rest*. Top-2 reduces the damage to 1/3 of Top-1 → the information lives in
> the tail, not the peak. **Deterministic in the max weight ≠ replaceable by selection.**

**Positive findings (despite refuting H1):**
1. The ΔPPL(L) curve is **smooth and monotonic, with no abrupt elbow** → there is NO first-order
   freezing point; the layers form a functional continuum, not separable discrete phases.
   (Refutes the naive picture of "discrete liquid/critical/frozen phases".)
2. The only abrupt transition is at **L0→L1** (Top-1: +523→+2308): L0 is qualitatively distinct
   (the only liquid one). The liquid/solid boundary exists but is at the BEGINNING, not the end.
3. Top-2 ≫ Top-1 systematically → a "tail integration" mechanism, not a "selection" one.

**Implication for the paper:** the "crystallization → acceleration" angle does not work via hard
top-k. But the thermodynamic characterization (Q01) + the quantitative refutation of hard selection
(Q02) + the functional continuum do tell an honest story about why attention is NOT compressible by
selection despite looking deterministic.

### GAP-L — Crystallized attention with residual excitations (J.P. Chancay, 2026-06-25)
Corrects the angle of Q02. The decomposition is **exact**:
$$\text{Attn} = \underbrace{V_{i^*}}_{\text{crystallized}} + \underbrace{\epsilon}_{\text{excitations}},
\qquad \epsilon = \sum_{i\neq i^*} p_i V_i$$
Q02 showed that $\epsilon$ is NOT negligible. The new question: **is $\epsilon$ compressible?**

**Correct physical interpretation (not "imaginary part"):** this is *ground state +
excitations*, $|\psi\rangle = |0\rangle + \epsilon|1\rangle+\dots$, NOT $a+ib$. The imaginary part
in QM has the same hierarchy as the real one and produces interference — it is not a corrective
term. The residual IS subdominant. So the rigorous analogy is ground-state + excitations; the
complex stays as representational inspiration, not a physical claim. (Anti-NQP discipline: do not
force the analogy.)

**EXP-Q03 — three replacement models in deep layers:**
- **A:** pure Top-1 (`V_{i*}`) — already measured in Q02 (breaks).
- **B:** Top-1 + full residual `V_{i*} + ε` — must recover exactly (sanity).
- **C:** Top-1 + low-rank residual `V_{i*} + λ·r`, r of dimension d_r ≪ d_head — the real test.
- **Metrics:** ΔPPL vs d_r. If a small d_r recovers almost all the loss → deep layers are *nearly
  crystalline with few excitations* that carry the information. Strong result.
- **Physics bonus:** |z|²=‖V_{i*}‖²+‖ε‖² and θ=atan(‖ε‖/‖V_{i*}‖) as an indicator of proximity to
  a transition / residual uncertainty.

### Result EXP-Q03 (2026-06-25) — Hε REFUTED (residual is NOT low-rank)
Sanity OK (B = exact baseline, ΔPPL=0). But the residual ε is not compressible:

| effective dim ε (90% var) | r=1 | r=4 | r=16 | r=32 |
|---|---|---|---|---|
| **21 / 64** | 33% rec | 46% | 71% | 87% |

The residual needs ~21/64 dims for 90% of variance; not even r=32 (half the space) recovers >87% of
the Top-1 loss. The ΔPPL(r) curve is smooth, with no early saturation. **The effective-theory
analogy (few excitation modes) does NOT apply:** the excitations are high-rank, closer to
*equipartitioned thermal noise* than to *few-mode phonons*.

### Unified conclusion of the crystallization line (Q02 + Q03)
**GPT-2's attention resists compression by every structural route tested:**
- Q02: not by hard selection (Top-k) → the tail of small weights does systematic work.
- Q03: not by low-rank projection of the residual → the tail is high-rank (~21/64).

The information of the attention "tail" is **irreducibly distributed** — dense, not peak +
correction. A strong negative result: against the intuition that "deterministic" heads (R≈0,
S_vn≈0) are compressible, the integration over context is essential and linearly incompressible.
(Possible escape: NONlinear compression of the residual, but already out of CPU scope.)

### Prioritization and target paper (J.P. Chancay, 2026-06-25)
Balance: 1 robust positive (thermodynamic structure) + a moderate signal (S_vn↔uncertainty) + 2
strong negatives (Q02 selection, Q03 low-rank). In AI architecture, **well-established negatives are
very valuable: they eliminate whole lines.** Priorities:
1. **S_vn as an uncertainty estimator** (low risk/cost, the signal already exists).
2. **Scaling law of L_c** (medium, reuses what's done — EXP-Q04 ready).
3. **Geometry of the residual tail (Dir-D):** PCA measures LINEAR rank; we are missing the
   **intrinsic dimension** (Isomap/MLE/TwoNN). If PCA→21 but intrinsic→5, the Q03 conclusion
   changes from "irreducible" to "nonlinear but compressible". **A necessary control to avoid
   overclaiming.**
- On hold: Top-k compression, deterministic replacements, strong quantum analogies.

**Target paper:** *"Thermodynamic Structure of Transformer Attention: Phase Separation, Uncertainty
Estimation, and the Irreducibility of Residual Context Integration"*. Narrative: (1) real
thermodynamic structure; (2) S_vn tracks uncertainty; (3) "crystallized" layers are NOT compressible
by simple linear approximations; (4) residual context integration looks like a fundamental property
of attention.

### EXP-Q05 (Dir-D) — intrinsic dimension of the residual → REFRAMES Q03
Estimate the intrinsic dim of ε (TwoNN/MLE, nonlinear) vs the 21 linear ones of PCA. It qualifies
whether the "irreducibility" of Q03 is linear-only or fundamental. Cheap (only over ε, already
collectible).

**Result (2026-06-25) — the tail DOES have low-dimensional (nonlinear) structure:**

| | linear PCA (90% var) | intrinsic dim (TwoNN) |
|---|---|---|
| residual ε (L9-11) | **23 / 64** | **6.8 / 64** |

(TwoNN validated vs swiss-roll: detects 2.7 where PCA sees 3 → distinguishes nonlinear from linear
rank.)

**Corrects the Q03 conclusion:** the tail is NOT "irreducible" — it is **LINEARLY irreducible but
lives on a ~7D nonlinear manifold**. The previous claim was an overclaim (linear-only). Correct
claim: *context integration is nonlinear but low-dimensional*. **It reopens the compression** that
Q02/Q03 seemed to close: the correct mechanism is a NONlinear encoder of the tail (e.g. MLP
64→~8→64), not linear truncation/projection.

**Lesson (anti-NQP discipline):** PCA measures linear rank, not dimension. "Not compressible by
PCA" ≠ "not compressible". Dir-D (J.P. Chancay) caught the overclaim before the paper. The paper
must say "linearly irreducible; low-dim nonlinear structure" — a more precise and in fact more
interesting claim (it suggests a mechanism, not just a limit).

### Evidence order (J.P. Chancay, 2026-06-25): Q04-lite BEFORE Q06
A methodological, not computational, reason: Q06 is mechanistic (can it be exploited?), Q04 is
external validity (is the phenomenon robust to scale?). The solid chaining for a paper is:
**the phenomenon exists → it scales → it has geometric structure → it can be exploited.**

**Central hypothesis (the highest-value bet):** $\dim(M_\epsilon) \approx \text{const}$ — the
intrinsic dimension of the contextual residual is ~independent of model size. If true, it is an
**effective reduction of degrees of freedom** (analogous to effective theories in physics), much
deeper and more general than a compression method. Scenarios:
- A: universal crystallization + dim_int≈const → Q06 absolute priority + a deep result.
- B: dim_int grows slowly → Q06 still makes sense.
- C: the phenomenon disappears in large models → Q06 loses interest.

### EXP-Q04-lite — phenomenon + scale (inference only)
For gpt2 small/medium/(large): measure L_c (via per-layer R) AND the **intrinsic dimension of the
residual** (TwoNN, the key observable). All forward-only. Discriminates A/B/C.

### Result EXP-Q04-lite (2026-06-25) — dim(M_ε) ≈ CONST (robust, Case A)
Controlled protocol (J.P. Chancay): identical N=1500 pts/head, 8 heads, last 3 layers, same
dataset, on gpt2 / medium / large (124M→355M→774M, 12→24→36 layers).

| model | layers | dim_int | dim_lin (PCA) |
|---|---|---|---|
| gpt2 | 12 | 7.2 ± 1.1 | 31.6 |
| gpt2-medium | 24 | 8.1 ± 0.8 | 30.5 |
| gpt2-large | 36 | 7.4 ± 0.7 | 28.0 |

**between-model spread = 0.9 ≈ within-model between-head noise = 0.8.** → The variation with scale
is indistinguishable from natural noise. The model grows 6× in params, 3× in layers, and the
**intrinsic dimension of the contextual residual stays at ~7-8.** A geometric effective core nearly
invariant to scale (J.P.'s Case A) — an effective reduction of degrees of freedom, analogous to an
effective theory. (Contrast: LINEAR dim ~30 vs intrinsic ~7 in all three → low-dim nonlinear
structure is universal, not a peculiarity of small.)

**Honesty — L_c is NOT universal:** L_c = 2/1/9 ("MIXED"). large has an extended liquid phase
(R: 0.86→…→0 gradual) vs small/medium that crystallize at L1-L2. The crystallization depth does
NOT give a clean law; **the universal observable is dim(M_ε), not L_c.** State it that way in the
paper.

**This is the project's strongest result** and survives the protocol control (unlike the first run
with uncontrolled N that gave an identical 6.7 = an artifact). It measures external validity: the
geometric phenomenon is robust to scale.

### EXP-Q06 (next) — nonlinear compression of the tail
Autoencoder 64→32→16→~8→16→32→64 over ε; measure reconstruction and then replace ε→ε̂ in the
Transformer, measure ΔPPL. Case A (ΔPPL≈0): tail nonlinearly compressible (a strong result).
Case B (degrades): geometric dim ≠ functional dim (also valuable). Methodological danger (J.P.):
TwoNN gives a LOCAL intrinsic dim; it does not guarantee a smooth global AE of that dim. That is
why Q06 FALSIFIES the H-NL hypothesis rather than assuming it.

### GAP-M — Is crystallization universal or contingent? Scaling law of L_c (J.P. Chancay)
**Formal question:** is the crystallization depth L_c a universal property of depth, or contingent
on architecture/size? Scenarios:
1. **Scales with depth:** L_c grows with N_layers (deep models → more crystalline region).
2. **Fixed point (RG):** L_c constant; adding layers replicates the crystalline regime (→ GAP-I).
3. **Partial recrystallization:** liquid→crystal→mixture→crystal (induction/recall heads require
   different regimes).
4. **Disappears at large scale:** GPT-2 small crystallizes due to low capacity ("hard"
   decisions); large models keep mixing → a small-model phenomenon, not universal.

**Strong result sought:** a law L_c = f(N_params, N_layers, N_data) — e.g. L_c ∝ log N, ∝ √N, or
constant. Any of these would be a scientific result on its own, and would tell whether "ground
state + excitations" is general or specific to GPT-2 small.

**EXP-Q04 (scale):** repeat the thermodynamic profile (Q01: T_eff, S_vn, per-layer R — CHEAP,
forward only) on GPT-2 small/medium/large. Locate L_c (the layer where R drops below threshold) in
each and look at the trend. Compute constraint (CPU): Q01 viable up to large; Q02/Q03 (dozens of
full PPLs) only on small. Strategy: L_c via the cheap Q01 signature on 3 sizes → trend.
