# Ablation design — what indexes the O_h universality class?

**Status:** ✅ **APPROVED (2026-06-26)** — pre-registered, vetted by two adversarial cross-check
rounds, approved by the PI. Ready for development (training harness). · **Author:** Juan Pablo Chancay
**Depends on:** cross_architecture_plan.md (Case B + 4 controls), the GAP-I (RG) frame in
theory/quantum_transformer_map.md
**Decision (2026-06-26):** mini matched-scale + RG-as-falsifiable-hypothesis (both recommended).
**Cross-check adjustments (2026-06-26):** (i) RG language softened to *fixed-point-like*, NOT
"universality classes" (insufficient evidence for a theory space + flows + identified relevant
operator); (ii) added Gate 0 (atlas-maturation) — a flat O_h from an immature model is an INVALID
experiment, not a refutation; (iii) A/B/C kept explicitly open — d_head may *parametrize/index* O_h
rather than *cause* it.

---

## 0. Why this experiment, and why now

Four measurement-only controls established:

1. The atlas exists across 4 architectures (Case B); O_h ≪ 1 universal.
2. O_h clusters by d_head (64 → ~0.28, 128 → ~0.20); the gap is real geometry (sweep, robustness).
3. **d_head is confounded with the *plateau* (deep-régime) intrinsic dimension** (intra-model
   ρ=−0.53 in Qwen).
4. **Model scale is NOT the lever:** with d_head fixed (GPT-2 family), O_h and plateau-d_int are
   flat; only peak-d_int grows. So O_h tracks the plateau d_int, which is set by d_head — not by size.

What measurement cannot do: break the d_head ↔ plateau-d_int confound, because in *pre-trained*
models the two always co-vary. The only way to separate them is to **train models that vary d_head
while holding everything else fixed**. This is the first step that crosses from measuring existing
models to training new ones.

## 1. The physics frame (falsifiable, not decorative)

The signature from control (4) — a quantity approximately invariant to scale (plateau d_int) plus a
quantity that flows with scale (peak d_int) — is *compatible with an RG language*: in RG,
scale-invariant quantities are fixed-point quantities, and the non-invariant ones are non-universal
(microscopic) details. Recent work formalizes RG for deep networks (arXiv:2510.25553).

**Deliberate restraint on the RG claim (anti-NQP, vetted).** We use **"fixed-point-like behavior"**,
*not* "universality classes". A universality class implies a space of theories, RG flows, stability
under perturbations, and an *identified relevant operator* — we have, today, two observables and two
regimes. That is far less. Claiming "distinct universality classes" from this would repeat the NQP
error (reifying an analogy: there it was Fisher = Hamiltonian; here it would be two-regimes =
universality classes). The fixed-point *language* is a lens, not a demonstrated structure.

**Pre-registered hypothesis (stated in the weaker, defensible form):**

> *d_head parametrizes a fixed-point-like O_h: within a fixed d_head, O_h is approximately invariant
> to scale; changing d_head shifts O_h to a different plateau. Whether d_head **causes** this or
> merely **indexes** a deeper geometric quantity is left open (see §2, hypotheses A/B/C).*

The ablation is the search for which architectural knob, under intervention, *moves* O_h — the
empirical content, independent of whether we ultimately call that knob "relevant" in the strict RG
sense.

**Anti-NQP guard.** This is a *prediction*, not a story: the experiment can refute it. If O_h moves
when d_head is held fixed but another knob varies, d_head is not the (sole) controlling axis. If O_h
does NOT move when d_head varies, the lead is wrong. We commit to §4 before training — *but only if
Gate 0 (§3.5) confirms the models are mature enough for a flat O_h to mean anything*.

## 2. The confound, stated as a causal diagram

We are choosing between:

```
   A:  d_head ─────────────► O_h            (d_head relevant; d_int a co-readout)
   B:  d_head ──► plateau-d_int ──► O_h      (d_int mediates; d_head proxy)
   C:  Λ ──► {d_head, plateau-d_int, O_h}    (latent class index; both are readouts)
```

Measurement could not separate these (d_head and d_int co-vary in pretrained models). Training breaks
the symmetry: we set d_head as an *intervention* and read plateau-d_int *post-hoc*.

**Causal-inference caveat (literature-confirmed).** plateau-d_int is a **post-treatment mediator**:
it is produced by the same training that d_head changed. A formal mediation analysis
(d_head→d_int→O_h) would be biased (violates sequential ignorability). So we treat plateau-d_int as an
**exploratory descriptive readout**, never as a causal mediator estimate. What the ablation *can*
claim cleanly is the **total effect** of the d_head intervention on O_h (A vs "d_head irrelevant"),
not the decomposition into direct/indirect paths.

## 3. Design — mini matched-scale, vary d_head only

Train small decoder-only LMs from scratch on WikiText-103, identical in every axis but d_head.

**Run-0 finding (2026-06-26) — depth raised 8 → 12.** The first attempt used n_layers = 8. It
*passed* G0a (the LM converged, val_loss ≈ 6.0) but *failed* G0b: plateau-d_int stuck at ≈ 4 across
all of training (vs ≈ 7–8 in pretrained GPT-2), i.e. no expansion→compression depth régime formed.
Gate 0 correctly flagged this as INVALID (under-depth), not a refutation — the anti-NQP gate doing its
job: O_h was a flat ≈ 0.40 for d_head = 32, which without the gate we might have read as a result. We
raise depth to **n_layers = 12** (= GPT-2, where the atlas and the phase profile demonstrably exist).
Matched-scale preserved: all four d_head variants are 63.7M params at 12 layers.

**Run-1 protocol revisions (2026-06-27) — recorded for faithful result comparison.** After run-1
(d_head=32 ×2 valid, d_head=64 s42 produced O_h=0.275 but FAILED G0b), a Phase-1 re-read of
Valeriani et al. (2302.00294) prompted two corrections. **Both runs (run-0/run-1) and any earlier
deep-layer numbers must be compared with these revisions in mind — they change the gate criterion and
the measurement depth.**

1. **G0b criterion: `bump_vs_ends` → `bump_vs_min` + early-peak check.** The original gate scored the
   depth régime as peak − max(endpoints). Valeriani report a *final ascent* phase (ID rises again near
   the last layers, "returning toward input-level values") — a HEALTHY part of the régime. A high
   last-layer ID shrank peak−ends and produced a **false negative**: d_head=64 failed with
   bump_vs_ends=0.18 while its bump_vs_min=1.44 was essentially identical to the d_head=32 runs that
   passed (1.54, 1.51). New criterion: **bump_vs_min > 0.5** (peak above the profile MINIMUM = the
   true régime amplitude, immune to the final ascent) **AND peak_rel ≤ 0.5** (Valeriani's early-peak
   finding: the ID peak sits in the first third/half). *Offline re-evaluation of the three run-1 JSONs
   confirmed: d_head=32 ×2 stay valid, d_head=64 s42 becomes valid (O_h=0.275). This is NOT loosening
   the gate arbitrarily — it aligns G0b with the régime the literature describes.*

2. **Measurement depth: rel 0.9 → rel 0.5 (Valeriani's compressed plateau).** Valeriani locate the
   minimum-ID / semantic plateau — the cleanest geometric régime — at rel ~0.4–0.5, NOT at rel ~0.9
   (their *final ascent*). The ablation now measures O_h and plateau-d_int in a 3-layer window centred
   on **rel 0.5**. **PROTOCOL DIVERGENCE (must not be silently mixed):** the cross-architecture results
   and all four controls measured O_h in the *deepest* layers (rel ~0.9). The ablation's plateau-centred
   O_h is therefore on a DIFFERENT basis; ablation numbers are compared to each other (across d_head),
   not 1:1 to the pretrained deep-layer 0.28/0.20 clusters. (The within-ablation contrast across d_head
   is what P1 tests, and it is internally consistent.)

**Fixed across all models (the matched-scale constraint):**
- d_model = 512, n_layers = 12, FFN = 4·d_model, context = 256, tokenizer = GPT-2 BPE
- data = WikiText-103 train, identical token budget, identical optimizer/schedule/seed-set
- positional = learned (one variant set) — so RoPE is NOT varied in this first batch
- norm = LayerNorm (one variant set) — RMSNorm not varied in this first batch

**The single varied axis — d_head (with n_heads adjusted to keep d_model constant):**

| variant | d_head | n_heads (= d_model/d_head) | note |
|---|---|---|---|
| H1 | 32  | 16 | below the 64 cluster |
| H2 | 64  | 8  | the "0.28 cluster" point |
| H3 | 128 | 4  | the "0.20 cluster" point |
| H4 | 256 | 2  | beyond the 128 cluster (extrapolation) |

**What this intervention is (precise statement, vetted).** d_model, depth, params (≈), data, and
optimization are identical; only the head partition of the same d_model changes. But d_head and
n_heads are **varied as a package**: at fixed d_model, n_heads = d_model/d_head is forced. So the
honest claim P1 can support is *not* "d_head is the relevant operator" but:

> *the intervention on attention-head geometry, implemented along the (d_head, n_heads) axis,
> produces a systematic change in O_h.*

Separating d_head from n_heads requires a **factorial** design (e.g. {d_model 512/768} × {d_head
64/128} gives same-d_head/different-n_heads and same-n_heads/different-d_head cells). We deliberately
do **not** do that now: we do not yet know the effect even exists under intervention. Batch-2
(factorial, fixing d_head = 64 and varying n_heads via d_model) runs **only if P1 is strong**.

**Two-seed replication** per variant (seeds 42, 123) for the same robustness discipline as the
cross-arch work.

**Compute and training budget.** 8-layer/512-wide models are ~15–30M params; 4 variants × 2 seeds =
8 short runs on a Colab T4. Priority order for the budget, per the cross-check: **more training steps
> more parameters > more data.** The dominant failure mode here is *under-training* (deep régime never
matures → false-negative flat O_h), not *under-sizing* — so we spend on steps first. Data stays
WikiText-103 (we want relative comparison and speed, not SOTA).

## 3.5 Gate 0 — atlas maturation (validity gate, BEFORE any prediction is tested)

The most dangerous failure mode of a from-scratch mini-batch is **under-training**: a model whose
deep régime has not matured will show a flat or degenerate O_h *for reasons unrelated to d_head*.
Reading that as "P1 refuted" would be a false negative. So before measuring O_h for the predictions,
each trained model must pass a maturation gate. **If a model fails Gate 0, it is an INVALID
measurement, not a refutation** — we extend training or exclude that model; we do NOT count it as
evidence about d_head.

Gate 0 checks (per model, on the held-out set):

- **G0a — converged LM.** Validation loss has plateaued (not still steeply descending); the model is
  a usable LM, not a half-trained one.
- **G0b — depth régime exists.** The per-layer intrinsic-dimension profile shows the
  expansion→compression→ascent shape (a deep plateau exists at all), as in pretrained GPT-2 — checked
  with `atlas_dhead_control`'s profile.
- **G0c — residual is stable.** The per-head residual ε is collectible with a stable SVD frame
  (sufficient effective rank, N adequate) — the same precondition every prior O_h measurement needed.
- **G0d — base atlas appears.** O_h ≪ 1 at all (i.e. heads are non-aligned at *some* level). This is
  P4, but here used as a *validity floor*: if even the existence of non-alignment fails to appear, the
  model is too immature to test magnitude on.

Only models passing G0a–G0d enter §4. This gate is what lets a flat O_h in §4 mean "d_head does not
move O_h" rather than "the model never formed an atlas".

## 4. Pre-registered predictions (commit BEFORE training)

Measured per model with the existing pipeline (`atlas_crossarch` inter-head O_h at fixed d_local=7,
deepest layers; `atlas_dhead_control` plateau d_int at relative depth 0.9):

- **P1 (the intervention test).** O_h decreases monotonically with d_head across H1→H4, tracking the
  cross-arch clusters (expect H2 ≈ 0.28, H3 ≈ 0.20). What a positive P1 establishes: *the (d_head,
  n_heads) intervention systematically moves O_h* — not yet "d_head is the relevant operator". If O_h
  is *flat* in d_head (and Gate 0 passed) → the (d_head, n_heads) axis does not move O_h; the lead
  collapses.
- **P2 (plateau-d_int co-moves).** plateau-d_int **changes systematically** with d_head (the
  within-pretrained negative d_int↔O_h relation reappears across trained variants). We pre-register
  *systematic*, **not monotonic** — a non-monotone but structured dependence (e.g. 32→6, 64→7,
  128→10, 256→9) still kills the simple-monotone story while leaving the phenomenon intact and
  interesting. Monotonicity is pre-registered only for O_h (P1), where the cross-arch sweep already
  showed it. If plateau-d_int is *flat* while O_h moves → d_int is NOT the mediator; A (direct) gains;
  C weakens. (Descriptive only — post-treatment mediator, §2.)
- **P3 (scale-invariance at fixed d_head — the most important control of the batch).** Two runs at
  fixed d_head = 64, n_layers = 12, varying scale via **d_model = 512 vs 768** (×2 seeds), measured at
  rel 0.5 like the rest of the ablation. P3 holds if the d_model=768 O_h reproduces the d_model=512 O_h
  (≈ 0.278, the Day-1 cluster-64 value) within the cross-arch robustness band (< 0.02). This is the
  cleanest evidence the design can produce: invariance to scale under a concrete intervention =
  **fixed-point-like behavior**. If O_h jumps with scale at fixed d_head → not a clean effective fixed
  point; the régime index is not d_head alone.
  **Covariate note (2026-06-27):** varying d_model at fixed d_head also changes n_head (512→8 heads,
  768→12 heads). P3 therefore tests scale with n_head as a *covariate*, not in isolation. This is
  acceptable because the cross-arch work already showed cluster-64 O_h ≈ 0.28 is robust to n_head
  (GPT-2 n=8 → 0.283, Qwen n=14 → 0.290); n_head is a known-robust covariate here, not a free
  confounder. Recorded so "fixed-point-like" is not over-read as "scale-pure". Run config:
  `scale_control_variants()` (d_model 512 & 768, n_layer 12).
- **P4 (existence is régime-independent).** Every variant has O_h ≪ 1 (the atlas exists at every
  d_head). A refutation here (some d_head gives O_h ≈ 1, aligned heads) would be the most surprising
  and important outcome. (Note: P4 also serves as Gate 0d, the validity floor.)
- **P5 (temporal emergence — what forms first, the atlas or the plateau? — added 2026-06-26).**
  Training from scratch gives a *direction* that frozen pretrained models cannot: we snapshot each
  run at 10/25/50/75/100 % of training and measure O_h(t) and plateau-d_int(t) at every snapshot.
  Pre-registered question: **does the plateau-d_int mature BEFORE O_h organizes, or after, or
  together?** If plateau-d_int reaches its final value while O_h is still high and only then O_h
  drops → temporal-ordering evidence for d_int → O_h that no frozen model can provide. If O_h
  organizes first → the reverse. We pre-register only that we will *report the ordering of the two
  emergence curves*; we do NOT pre-commit to which comes first (genuinely open). This is a directional
  hint, NOT a causal proof (training is one coupled process), and the atlas's temporal emergence may
  be a result in its own right.

**Verdict matrix.** P1∧P2 → B/C (d_head moves O_h via the plateau-d_int it induces). P1∧¬P2 → A
(direct). ¬P1 (Gate 0 passed) → the (d_head, n_heads) axis is not the lever — re-examine. P3 is the
fixed-point-like gate; P4 is the existence floor. Throughout, "d_head moves O_h" stays distinct from
"d_head *causes* O_h": the package may **index/parametrize** a deeper geometric quantity (C) without
being the mechanism — a possibility we keep explicitly alive.

## 4.5 Post-hoc observations from run-1 (EXPLORATORY — NOT pre-registered)

These two patterns were *noticed in the run-1 data*, AFTER seeing it (d_head=32, rel-0.5). They are
**not** pre-registered predictions and must not be reported with the same strength as P1–P5. They are
**candidates for a NEW AICR cycle** — and per the method, an interesting post-hoc pattern earns a
hypothesis only after passing search → second opinion. The agreed flow before testing either:
**(1) web/literature search → (2) adversarial second opinion → (3) only then form a hypothesis and
pre-register a test.** Recorded here so they are not lost and not silently promoted.

- **OBS-A (temporal emergence order).** In run-1, O_h is essentially flat from ~10 % of training
  (≈ 0.40) while plateau-d_int is still moving (rises to a mid-training peak, then decays toward the
  end). *If* this holds across d_head, the emergence ORDER (O_h early, plateau-d_int late) would
  argue **against d_int → O_h** (a cause cannot arrive after its effect) and **toward hypothesis C**
  (a latent Λ fixed early, with O_h a fast readout and d_int a slower, separate dynamic). Note the
  rise-then-decay of plateau-d_int *in training time* mirrors Valeriani's expansion→compression
  *in depth* — possibly a finding in its own right. **Do not conclude from one d_head.**
- **OBS-B (plateau-d_int is seed-noisy; O_h is seed-robust).** Between the two d_head=32 seeds, O_h
  differed by 0.001 while plateau-d_int differed by up to ~1.2 at mid-training. *If* general, this
  says O_h is a "hard" (stable, reproducible) quantity and plateau-d_int a "soft" (init-sensitive,
  dynamic) one — i.e. they may sit at different causal levels (O_h more fundamental, d_int more
  emergent), again consistent with C. **Methodological consequence for P2:** for plateau-d_int to
  count as "changing with d_head", the between-d_head difference must clearly exceed the ~1.2
  within-d_head seed spread — the same within-vs-between discipline used in the cross-arch work.

- **OBS-C (temporal location of the plateau-d_int peak; the "O_h peak" is NOISE).** Noticed that the
  plateau-d_int *peak* seems to occur at different training fractions across runs (d_head=64 s123 peaks
  ~25 %, d_head=128 s42 peaks ~50 %). **Important caveat established by the analyst:** the apparent
  "O_h peak" that prompted this is NOT real — O_h varies by ±0.004–0.009 across snapshots, the size of
  the N=600 measurement noise, so O_h is flat-plus-noise with no peak (correlating O_h's noise with
  d_int's real peak is a spurious-coincidence trap, correctly rejected). What MAY be real: the
  plateau-d_int peak location varies — but with 1 seed per high-d_head and d_int being seed-noisy
  (OBS-B), this cannot separate "peak moves with d_head" from "peak moves with seed". Requires both
  seeds × several d_head. Folds into the H-TEMP line as a sub-question.

**Status (2026-06-27):** AICR cycle for OBS-A/B advanced through **Phase 1 (search) + Phase 3 (second
opinion)**; pre-registration + test (Phase 4+) **deferred until the main ablation closes** (P1 with
128/256, then P3) — keep focus on the pre-registered lines first. Captured below so the cycle resumes
cold.

**Phase-1 findings (literature).**
- Functional attention structure (induction/retrieval heads) emerges *gradually*, in stages, with
  loss phase-changes (2502.06902, 2411.12118, 2404.07129). This is **function**, a different object
  from our **subspace geometry** — so it does NOT contradict OBS-A (O_h flat early).
- Attention-output low-rankness is *partly architectural* — induced by W^O and the constraint
  dim(⋃ span(headᵢ)) ≤ d_head·n_head (2508.16929). That paper EXPLICITLY does **not** test whether
  the structure exists at init or emerges in training — an open gap our O_h(t) touches.
- Intrinsic dim *rises-then-compresses during training* (Ansuini 1905.12784) — confirms OBS-A's
  plateau-d_int rise→peak→decay as a KNOWN phenomenon (not our novelty).
- Geometric quantities can have per-seed variance of different orders of magnitude (2601.13303,
  360–500× vs accuracy) — makes OBS-B plausible.

**Phase-3 (adversarial cross-check) — accepted points.**
- **Mandatory baseline control:** O_h≈0.40 at init could be TRIVIAL geometry induced by
  d_model/d_head/W^O with no functional structure. The test must produce THREE curves per d_head:
  O_h(init), O_h(randomized: gaussian same-cov / SVD-randomized / sample-permuted), O_h(trained).
  Without the random baseline, "O_h at init" is uninterpretable.
- **Claim is bounded:** the supported statement is **"O_h converges much earlier than d_int"**, NOT
  "O_h is architectural" — fast self-organization (symmetry breaking in the first hundreds of steps,
  LayerNorm bias) would also give early convergence without being architectural. The **step-0**
  measurement is what separates "present at init" from "rapidly acquired".
- OBS-B adds **independent convergent evidence** (a distinct qualitative prediction), and "robust vs
  constant-by-construction" is separated by an **init-sensitivity test** (Xavier/Kaiming/Orthogonal/
  gain): O_h constant across inits ⇒ architectural.
- Possible decomposition O_h = O_struct + O_learned (do NOT assume O_struct dominates — that is
  ChatGPT's prior, declared, not expected).

**Pre-registered hypothesis (drafted, to commit when the line opens) — H-TEMP.**
> *Inter-head overlap O_h converges significantly earlier than the residual intrinsic dimension. We
> test whether O_h is already present at initialization (vs a random-geometry baseline) or is
> rapidly acquired in the earliest optimization steps.*
Three tests: **T1** step-0 measurement + random baseline (no training — cheap, decisive);
**T2** full temporal evolution (already collected via P5); **T3** init-sensitivity. Secondary line,
not a new project. The current batch's `aggregate.py` already supplies the T2 raw evidence
(emergence order + per-d_head seed spread).

## 4.6 Batch-2 plan (decided 2026-06-29; DETAILED design deferred until batch-1 closes)

Batch-1 confirmed P1 (O_h 0.40→0.28→0.20 across d_head 32/64/128, two clusters reproduced from
scratch). Decision: run batch-2 **in two stages, cheap-then-expensive** (validate before spending).
Detailed pre-registration written only after batch-1 + P3 close — this is the skeleton.

- **Stage 1 — MORE SEEDS of batch-1 (cheap, consolidation).** Re-run the 4 d_head with **4–6 seeds**
  (vs 2). Tightens P1 CIs and, crucially, **resolves OBS-B with rigour**: is the s123 > s42
  plateau-d_int bias real (3/3 so far, p≈0.125) or chance? With 4–6 seeds the seed effect is
  measurable. Also tests OBS-C (does the d_int-peak location move with d_head, or is it seed-driven?).
  Same small models, just more runs → fits short GPU sessions, resume-safe.
- **Stage 2 — FACTORIAL d_head × n_head (expensive, disentangles).** A 2-D grid varying d_model
  (e.g. 512/768/1024) so that **d_head and n_head can be separated** — the question batch-1 cannot
  answer (they are locked at fixed d_model). Cells like {d_model 512, n_head 4, d_head 128} vs
  {d_model 1024, n_head 8, d_head 128} (same d_head, different n_head) vs {d_model 1024, n_head 16,
  d_head 64} (same n_head, different d_head). Answers: **is O_h moved by d_head or by n_head?** Run
  only if Stage 1 keeps P1 solid. Keep d_model modest (≤768–1024) to bound compute.

**On d_head=256 (n_head=2) — batch-1 result: Gate 0 FAILED (régime is degenerate, not just
under-founded).** Two problems are CONFOUNDED at d_model=512: (a) one inter-head pair (sampling — not
a population property; more seeds do NOT fix it), and (b) Gate 0 G0b failed — the depth profile is
anomalous (late peak at rel 0.6, not the early-peak expansion→compression régime; correctly flagged
INVALID by the Valeriani early-peak check). O_h also decays monotonically (0.15→0.13), breaking OBS-A.
With 2 heads there is no "population of subspaces" to organize → a degenerate régime.

**→ d_head=256 is now a PRIORITY case for the Stage-2 factorial** (not to be chased with more seeds).
The factorial disentangles the two confounded problems: {d_model 1024, n_head 4, d_head 256} (6 pairs)
and {d_model 2048, n_head 8, d_head 256} (28 pairs) test whether d_head=256 is pathological **because
it is d_head=256** or **because n_head=2**. If the atlas forms healthily at d_head=256 with 4–8 heads
and O_h follows the pendiente → the problem was n_head=2, and P1 extends cleanly to a 4th point. If it
stays pathological with more heads → d_head=256 is a genuinely distinct régime. **Enter neutral: the
factorial TESTS the cause, it does not aim to "rescue" the point** (entering to rescue would bias the
read — anti-overclaim). Both outcomes are valid results.

**Batch-1 P1 stands on the 3 VALID points (32/64/128).** d_head=256 exits §4 as INVALID via Gate 0,
not as a curve point — the gate again excluding a spurious 4th point (a degenerate régime) exactly as
designed. Observation worth keeping: *the atlas may require ≥4 heads to form a healthy depth régime.*

## 5. What this does and does NOT establish (scope, anti-NQP)

- **Does:** the *total effect* of the (d_head, n_heads) intervention on O_h, at matched scale, on
  trained-from-scratch models — the first non-correlational statement in the project. "Total effect"
  in the causal-inference sense: the change in O_h caused by the intervention, NOT decomposed into
  direct/indirect paths (the mediation decomposition is blocked by the post-treatment mediator, §2).
- **Does NOT — parametrize ≠ cause.** Even a clean P1∧P3 shows the intervention *moves/indexes* O_h,
  not that d_head is the deep mechanism. In complex systems a variable often organizes/indexes/
  correlates with an observable without being its cause; d_head may simply be the first hyperparameter
  found that aligns with a deeper residual geometry (hypothesis C). We will say "d_head parametrizes a
  fixed-point-like O_h", not "d_head causes the atlas".
- **Does NOT:** O_h → model quality (a separate interventional question; needs a downstream-task
  comparison and is explicitly out of scope here). Conflating the two would be the NQP error in its
  purest form.
- **Does NOT (yet):** isolate d_head from n_heads (the package, §3), nor from RoPE/RMSNorm (held
  fixed). Those are batch-2 axes, run only if P1 holds and Gate 0 passed.

## 6. New scopes the physics frame opens (beyond this ablation)

Recorded for the project's medium-term map; these are *derivable directions*, not commitments:

- **Régime selection as a design tool.** If d_head parametrizes a fixed-point-like O_h, then "choose
  the attention régime" becomes a first-class design decision — pick d_head for a target O_h (head
  specialization budget), not just for compute. A geometric, not heuristic, basis for the 64–128
  folklore. (The stronger "universality class" framing is deferred until — if ever — RG flows and a
  relevant operator are actually demonstrated, not just observed.)
- **The unexplored physical GAPs that may interact with O_h** (from quantum_transformer_map.md):
  GAP-G (mutual information between heads — long-range correlation = criticality indicator; directly
  complements O_h), GAP-I (RG coarse-graining across depth — the frame we are now using empirically),
  GAP-K (heat capacity peaks = regime changes). O_h may be one coordinate of a richer
  "geometric phase diagram" of attention.
- **The reframed central question** (from cross_architecture_plan.md §"Reframed research question"):
  the minimal geometric quantity Λ that jointly organizes plateau-d_int and O_h. If P3 holds, Λ is
  *consistent with* being a fixed-point-like quantity and d_head is a *candidate control parameter* for
  it — a first concrete handle on the hitherto-abstract Λ. (We stop short of "Λ is a fixed-point
  quantity" and "d_head is its relevant operator": P3 shows scale-invariance under one intervention,
  which is consistency, not a demonstrated RG fixed point or an identified relevant operator.)

## 7. Open decision before development

- Train/eval harness: minimal from-scratch GPT (nanoGPT-style) vs a `transformers`
  `GPT2Config`-driven training loop. The latter reuses our existing measurement backends directly
  (`residual_backends` already speaks GPT-2). **Recommendation: `transformers` GPT2Config**, varying
  only `n_head` (which sets d_head = n_embd/n_head) — zero new measurement code, the ablation models
  are read by the exact same pipeline that produced every prior result.
