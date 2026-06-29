---
name: nqp-session-state
description: Estado de la última sesión de trabajo en NQP/ATLAS
metadata:
  type: project
---

# Estado de sesión — NQP / ATLAS

**Última sesión**: 2026-06-26
**Rama activa**: atlas → ablation (EXP-ABLATION en ejecución)

## EXP-ABLATION — estado (2026-06-26, fin de sesión)

**Harness construido, auditado, resume-safe.** Estructura `src/ablation/` (train/measure/
experiments/results). Corre con `python run_batch1.py --mode batch1 --device cuda --steps 6000`.

**Calibración resuelta:**
- n_layer 8 → **12** (8 capas fallaban Gate 0 G0b: sin régimen de profundidad; 12 = GPT-2 lo da).
- fp16 autocast (cuda) activo: ~177s/600 pasos, ~29 min/run.
- Medición P5 ligera en intermedios (N=600), plena en final (N=1200).
- Matched-scale a 12 capas: 4 d_head → 63.7M idénticos.
- G0b mide FORMA del perfil (bump >0.5), NO magnitud de d_int. Instrumentado (imprime perfil)
  + fix bug NaN en endpoints.
- Resume-safe: `--resume` (default) salta runs con JSON existente; `--save-ckpt` opcional.

**ÚNICO punto medido (run perdido al morir runtime Colab, pero capturado en chat):**
- **d_head=32, 12 capas, seed=42 → O_h ≈ 0.401, Gate 0 PASÓ (válido).**
- Curva P5: O_h plano en ~0.40 desde el 10% del entrenamiento (val_loss 6.79→6.01);
  plateau_d_int 3.3→4.0. **El atlas emerge tempranísimo, no se refina al final.** (Dato P5 fuerte.)
- Consistente con P1: d_head=32 (< cluster 64) → O_h mayor (0.40 > 0.28). Falta confirmar
  64→~0.28 y 128→~0.20.

**REVISIÓN DE PROTOCOLO run-1 (2026-06-27) — CRÍTICO para comparar resultados.** Tras run-1
(d_head=32 ×2 válidos O_h~0.40; d_head=64 s42 dio O_h=0.275 pero FALLÓ G0b), re-lectura Valeriani
(Fase 1) → 2 correcciones, registradas en docs/ablation_design.md §"Run-1 protocol revisions":
- **G0b: bump_vs_ends → bump_vs_min + check pico temprano (rel≤0.5).** bump_vs_ends penalizaba el
  *ascenso final* de Valeriani (fase sana). d_head=64 falló con bump_vs_ends=0.18 pero bump_vs_min=1.44
  ≈ idéntico a los d_head=32 que pasaron (1.54,1.51). Re-evaluación offline: d_head=64 s42 → VÁLIDO,
  O_h=0.275. NO es aflojar el gate: lo alinea con el régimen que Valeriani describe.
- **Medición rel 0.9 → rel 0.5** (plateau comprimido de Valeriani, geometría más limpia; rel 0.9 era
  su *ascenso final*). DIVERGENCIA DE PROTOCOLO: cross-arch + 4 controles midieron en deep (rel~0.9);
  el ablation mide en rel 0.5. Los números del ablation se comparan ENTRE SÍ (across d_head), NO 1:1
  con los clusters pretrained 0.28/0.20. P1 testea el contraste interno, que es consistente.
- Añadido flag `--d-heads 64 128` para run dirigido.

**PUNTOS VÁLIDOS hasta ahora (con G0b corregido, medición rel 0.9 del run-1 — OJO: run-2 medirá rel 0.5):**
- d_head=32: O_h≈0.40 (×2 seeds) | d_head=64: O_h=0.275 (1 seed). Pendiente P1 decreciente: 0.40→0.28.
- **CUIDADO comparabilidad:** estos O_h del run-1 son rel 0.9; el run-2 (rel 0.5) puede dar valores
  distintos. Para P1 lo que importa es el contraste across d_head DENTRO de cada protocolo.

**BLOQUEO: límite GPU Colab (~12-24h).** Reanudar run-2 DIRIGIDO: Celda 1 (pull) → run con
`--d-heads 64 128` (faltan 64 s123, 128 ×2; 32 ya válidos). --resume salta lo hecho.

**Pendiente token HF**: revocar hf_BhgO... (expuesto en sesiones previas).

**CONVENCIÓN (decisión usuario 2026-06-27): un Colab NUEVO por test/etapa.** Cada corrida usa su
propio notebook → queda como registro inmutable de la prueba (output completo, qué se corrió, cuándo).
Coherente con la memoria persistente de AICR. Para el Día 2: notebook nuevo, NO reusar el del Día 1.


## Resultado central de esta sesión — arco cross-architecture CERRADO

Extendido el atlas de GPT-2 a 4 familias autorregresivas → **Case B**:
- O_h ≪ 1 universal (existencia robusta a la arquitectura).
- Magnitud clusteriza por diseño de atención: d_head 64 → ~0.28 (GPT-2/Qwen),
  d_head 128 → ~0.20 (Llama/Mistral). CIs no se solapan.
- Control d_local fijo (k=7): el gap **se ensancha** a ΔO_h ≈ 0.09 → geometría real,
  no artefacto de métrica. (Refuta el caveat anti-NQP sobre la métrica.)
- Sweep O_h(k) k=4..10 (GPT-2/Qwen completos; Llama/Mistral 2 puntos): los dos modelos
  d_head-64 trazan una **curva común** (cohesión ≤0.022); el cluster-128 está debajo en
  todo k (Mistral@k9=0.226 < Qwen@k9=0.325). El gap es un offset vertical de la curva, no
  artefacto de k=7. Datos: `docs/dlocal_sweep.json`.
- **Control intra-modelo (d_head vs d_int)**: dentro de un modelo, ρ(d_int_h, Ō_h).
  Qwen ρ=−0.53 p=3e-4 (sig.), GPT-2 ρ=−0.26 p=0.13 (mismo signo, n.s.). **d_head está
  CONFUNDIDO con d_int** (el vínculo es per-head en Qwen). El lead d_head baja a "principal
  sospechoso, confundido". El ablation debe trackear d_int como mediador.
  Código `src/atlas_intramodel.py`; datos `docs/intramodel_{gpt2,qwen}.json`.
- **Experimento barato — el tamaño NO es la palanca**: d_int es emergente (no hiperparámetro),
  así que la intervención conversa: d_head FIJO (=64, familia GPT-2), variar tamaño, a profundidad
  RELATIVA fija (0.9). Resultado: O_h plano (spread 0.002) y plateau d_int plano (0.15) en 12→36
  capas; solo el PEAK d_int crece (1.45) = predicción de Valeriani. → O_h sigue al *plateau* d_int,
  no al tamaño. El acoplamiento d_head↔plateau-d_int↔O_h NO es efecto de escala.
  Código `src/atlas_dhead_control.py`; datos `docs/dhead_control.json`.
- **Búsqueda web (vetting de ChatGPT)**: mediación post-tratamiento confirmada espuria (no va al
  paper); d_int-emergente confirmado (Valeriani 2302.00294); intuición "Λ latente" refinada al
  split concreto peak/plateau de d_int. La postura de ChatGPT sobrevivió el escrutinio.
- **Pregunta reformulada (norte nuevo)**: de "¿qué hiperparámetro fija O_h?" a "¿cuál es la cantidad
  geométrica mínima que organiza conjuntamente plateau-d_int y O_h, y qué decisiones arquitectónicas
  la modulan?". O_h↔plateau-d_int, fijado por d_head (entre clusters), no por escala.
- **Control de robustez (3 capas × 2 seeds, k=7)**: todos STABLE.
  Qwen depth/seed-spread 0.017/0.005 (mean 0.283), Mistral 0.006/0.002 (0.198),
  Llama 0.003/0.001 (0.196). Wobble máx (Qwen 0.017) ≈ 5× menor que el gap 0.08.
  Llama≈Mistral a 0.002 (misma geometría 32/8/4/128, distinto corpus).

## Cambios realizados en esta sesión

- `src/residual_backends.py` (nuevo): extracción ε architecture-agnostic (GPT-2/Llama/Mistral/Qwen2).
- `src/intrinsic.py`: `collect_residuals` ahora delega al backend.
- `src/atlas_crossarch.py` (nuevo + extendido): driver cross-arch con split GQA intra/inter,
  d_local sweep, control k=7, y **`robustness()`** (depth×seed, veredicto vs XARCH_GAP=0.08).
  Commit dab2d05 (robustness había quedado sin commitear; se pusheó esta sesión).
- `tests/test_phase0_regression.py` (nuevo): gate GPT-2 O_h=0.284 bit-for-bit.
- `docs/cross_architecture_plan.md`: registrados Phase 0/1/2 + d_local control + robustness.
- `docs/paper_draft.md` (editado con autorización; título promovido):
  §3.1b cross-arch + párrafo de robustez; Appendix E (protocolo GQA + tabla robustez); §6/§7.
- `src/make_figures.py`: `fig6_crossarch()` → docs/figures/fig6_crossarch.png.
- `README.md`: actualizado a "four autoregressive families" + Case B.
- `docs/phase2_results.json`, `docs/phase2_control.json`: datos cross-arch.
- `.gitignore`: token HF y archivos de secreto.

## Pendientes para próxima sesión

- **P-CHATGPT**: consultar a ChatGPT ANTES de diseñar el ablation arquitectónico (decisión del usuario).
- **EXP-ABLATION — DISEÑO APROBADO (2026-06-26)**: `docs/ablation_design.md`, vetted en 2 rondas
  con ChatGPT, aprobado por el PI. Batch-1 mini matched-scale: 4 modelos ~20M (d_model=512, 8 capas,
  LayerNorm, learned-pos), variar d_head ∈ {32,64,128,256} (n_heads=16/8/4/2 — paquete, NO se separa
  ahora), 2 seeds, WikiText-103, Colab T4. Frame: **fixed-point-like** (NO "universality classes").
  Gate 0 (atlas maduro) antes de medir: O_h plano sin atlas = INVÁLIDO, no refutado. P1 monótona en
  O_h; P2 "sistemática" (no monótona) en d_int; P3 (d_head=64, escala distinta → mismo O_h) = control
  clave. Afirma efecto TOTAL de intervención, NO causa, NO O_h→quality. Factorial (batch-2) solo si
  P1 fuerte.
  **SIGUIENTE = DESARROLLO**: harness de entrenamiento con transformers GP2Config variando solo
  n_head; los modelos los lee el mismo pipeline (residual_backends ya habla GPT-2); + Gate 0.
- **SEGURIDAD**: revocar el token HF expuesto en chat (hf_BhgO...) y generar uno nuevo.
- **P-001**: confirmación bibliográfica refs [17]–[23] (flagged [verify]) vía reference manager.
- Integrar §3.1b → ya hecho; solo queda re-render de figuras si cambian datos.
