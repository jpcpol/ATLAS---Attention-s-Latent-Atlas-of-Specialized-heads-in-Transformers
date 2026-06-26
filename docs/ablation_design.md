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

**Fixed across all models (the matched-scale constraint):**
- d_model = 512, n_layers = 8, FFN = 4·d_model, context = 256, tokenizer = GPT-2 BPE
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
- **P3 (scale-invariance at fixed d_head — the most important control of the batch).** A second
  mini-run at d_model = 768, n_layers = 8, fixed d_head = 64 (a different scale, same régime)
  reproduces H2's O_h within the cross-arch robustness band (< 0.02). This is the cleanest evidence
  the design can produce: invariance to scale under a concrete intervention = **fixed-point-like
  behavior**. If O_h jumps with scale at fixed d_head → not a clean effective fixed point; the régime
  index is not d_head alone.
- **P4 (existence is régime-independent).** Every variant has O_h ≪ 1 (the atlas exists at every
  d_head). A refutation here (some d_head gives O_h ≈ 1, aligned heads) would be the most surprising
  and important outcome. (Note: P4 also serves as Gate 0d, the validity floor.)

**Verdict matrix.** P1∧P2 → B/C (d_head moves O_h via the plateau-d_int it induces). P1∧¬P2 → A
(direct). ¬P1 (Gate 0 passed) → the (d_head, n_heads) axis is not the lever — re-examine. P3 is the
fixed-point-like gate; P4 is the existence floor. Throughout, "d_head moves O_h" stays distinct from
"d_head *causes* O_h": the package may **index/parametrize** a deeper geometric quantity (C) without
being the mechanism — a possibility we keep explicitly alive.

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
