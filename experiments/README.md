# NQP — Experiments

## Orden de ejecución propuesto

### EXP-001 — Baseline Fisher diagonal en GPT-2 small
**Objetivo:** verificar que $\hat{P}$ derivado de Fisher diagonal reduce $\varepsilon_Q$
respecto a cuantización estándar en un modelo controlado (GPT-2 124M).

**Métricas:** PPL en WikiText-103, error de cuantización L2 por capa, bits efectivos usados.

**Comparadores:** INT8 estándar, GPTQ 4-bit, QuIP 4-bit.

**Estado:** `src/fisher.py` implementado. Listo para ejecutar con `python src/fisher.py --bits 8 --n-calib 256`.

---

### EXP-002 — Escala a Llama-3 8B
**Prerequisito:** EXP-001 muestra mejora en GPT-2.

**Objetivo:** validar NQP-C1 a escala práctica.

**Estado:** pendiente.

---

### EXP-003 — Test de NQP-C2 (forma fuerte)
**Objetivo:** verificar si modelo NQP-4bit supera FP32 en tareas específicas.

**Hipótesis de trabajo:** en tareas de code review (distribución de calibración = código),
NQP actúa como regularizador y mejora precisión vs FP32.

**Estado:** pendiente — requiere EXP-002.
