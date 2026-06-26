---
name: nqp-session-state
description: Estado de la última sesión de trabajo en NQP/ATLAS
metadata:
  type: project
---

# Estado de sesión — NQP / ATLAS

**Última sesión**: 2026-06-26
**Rama activa**: atlas (cross-architecture)

## Resultado central de esta sesión — arco cross-architecture CERRADO

Extendido el atlas de GPT-2 a 4 familias autorregresivas → **Case B**:
- O_h ≪ 1 universal (existencia robusta a la arquitectura).
- Magnitud clusteriza por diseño de atención: d_head 64 → ~0.28 (GPT-2/Qwen),
  d_head 128 → ~0.20 (Llama/Mistral). CIs no se solapan.
- Control d_local fijo (k=7): el gap **se ensancha** a ΔO_h ≈ 0.09 → geometría real,
  no artefacto de métrica. (Refuta el caveat anti-NQP sobre la métrica.)
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
- **EXP-ABLATION**: ablation matched-scale {MHA↔GQA, #KV, d_head, RoPE↔learned, RMSNorm↔LN} —
  ¿qué componente fija O_h? d_head es el lead. Establece architecture→O_h, NO O_h→quality (lección NQP).
- **SEGURIDAD**: revocar el token HF expuesto en chat (hf_BhgO...) y generar uno nuevo.
- **P-001**: confirmación bibliográfica refs [17]–[23] (flagged [verify]) vía reference manager.
- Integrar §3.1b → ya hecho; solo queda re-render de figuras si cambian datos.
