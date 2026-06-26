---
name: nqp-research-decisions
description: Decisiones de investigación — pivots, metodología, lecciones
metadata:
  type: project
---

# Decisiones de investigación — NQP / ATLAS

## D-001: Pivot de cuantización a geometría de atención (2026-06-24)

**Decisión:** Al refutarse NQP-C1, no abandonar sino reorientar hacia la pregunta: "¿falla la analogía cuántica completamente, o hay alguna parte que sí describe el transformer?"

**Evidencia que motivó el pivot:** Fisher de activaciones es rank ~2 (no el operador rico que la intuición presupuso). Sin embargo, softmax = Boltzmann literalmente → línea termodynamics abre resultados positivos.

**Consecuencia:** El proyecto pasó de "cuantización" (aplicado) a "representación geométrica de atención" (interpretability).

---

## D-002: Mantener resultados negativos documentados (2026-06-24)

**Decisión:** No borrar EXP-001, A-G4, ni EXP-Q06. Documentar con el mismo rigor que los positivos.

**Razón:** La disciplina de reportar negativos es parte del valor científico del proyecto. El paper §3 los incluye explícitamente.

---

## D-003: Congelar paper_draft.md (2026-06-25)

**Decisión:** `docs/paper_draft.md` es frozen — no modificar sin instrucción explícita.

**Razón:** El preprint representa el estado del arte del proyecto en el momento en que el resultado central fue establecido. Modificaciones deberían ser una nueva versión, no edición in-place.

---

## D-004: Lección metodológica — naive tool over-concludes (2026-06-25)

**Patrón identificado 4 veces:** la herramienta naive sobre-concluye; el control correcto reencuadra.
1. L2 error biasa contra NQP
2. Correlación bivariada vs parcial en U1b
3. Rango lineal (PCA) vs intrínseco (TwoNN)
4. "Atlas" como fiber bundle formal vs descriptivo

**Aplicación futura:** antes de reportar un resultado, preguntar: ¿qué control puede cambiar el signo de esta conclusión?

---

## Invariantes del proyecto

1. `paper_draft.md` es frozen — no editar in-place
2. Resultados negativos se documentan con el mismo rigor que los positivos
3. El claim central (O_h ≈ 0.28, scale-invariant) no se debilita sin evidencia contraria robusta
4. La pregunta abierta central — ¿causal o descriptiva? — no tiene respuesta hasta P-002
