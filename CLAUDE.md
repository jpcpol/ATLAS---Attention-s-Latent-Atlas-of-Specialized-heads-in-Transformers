# NQP — Natural Quantization via State Preparation

## Propósito

Proyecto de investigación en optimización de cuantización de LLMs mediante transformaciones
que alinean la rejilla discreta con la geometría natural del espacio de pesos.

**Investigador principal:** Juan Pablo Chancay (jpcpol@gmail.com)
**Inicio:** 2026-06-24
**Venue objetivo:** NeurIPS / ICLR (TBD)
**Relación con otros proyectos:** NQP es independiente de CAL. Si NQP produce resultados,
puede usarse como infraestructura de deployment para los modelos que operan dentro de CAL L2.

---

## Intuición fundacional

En mecánica cuántica, un sistema en superposición colapsa al medir a un eigenvalor del
operador de medición — la discreción emerge de la geometría del sistema, no se le impone.

En cuantización ML estándar, la rejilla discreta se impone externamente sobre la distribución
de pesos, generando ruido de cuantización porque no respeta la geometría natural.

**Hipótesis NQP:** existe un operador de preparación $\hat{P}$ (derivado de la métrica de
Fisher del modelo) tal que cuantizar en la base de $\hat{P}$ minimiza el error de cuantización
— análogo a medir en la base propia del Hamiltoniano.

---

## Estructura del proyecto

```
NQP/
├── theory/
│   └── operator_formalization.md   ← formalización matemática del operador $\hat{P}$
├── experiments/                     ← scripts de validación empírica
├── src/                             ← implementación del operador NQP
└── docs/                            ← papers, presentaciones, referencias
```

---

## Conceptos clave

| Símbolo | Definición |
|---|---|
| $W \in \mathbb{R}^d$ | Vector de pesos de una capa |
| $F$ | Matriz de información de Fisher del modelo |
| $\hat{P} = U$ | Operador de preparación — diagonaliza $F$ |
| $\tilde{W} = \hat{P}W$ | Pesos en base natural |
| $Q_i$ | Cuantizador adaptativo por componente $i$ |
| $\lambda_i$ | Eigenvalor de Fisher — curvatura de loss en dirección $i$ |
| $\varepsilon_{\text{NQP}}$ | Error de cuantización en base natural |

---

## Conjeturas activas

- **NQP-C1:** $\hat{P}$ óptimo lleva el error de cuantización a $\varepsilon_{\text{NQP}} \ll \varepsilon_{\text{std}}$ para cualquier presupuesto de bits.
- **NQP-C2 (fuerte):** la cuantización en base natural con $b$ bits puede superar FP32 sin cuantización en tareas donde la distribución de calibración es representativa.

---

## Estado

| Componente | Estado |
|---|---|
| Formalización del operador $\hat{P}$ | 🟡 Borrador inicial |
| Implementación Fisher diagonal (GPT-2) | 🟢 Implementado en `src/fisher.py` |
| Validación empírica NQP-C1 | ⬜ Pendiente |
| Comparación vs GPTQ / QuIP# | ⬜ Pendiente |
| Paper draft | ⬜ Pendiente |

---

## Modelo LLM

Sin restricción de instrumento aquí (a diferencia de CAL/L2 donde φ está calibrado para
`claude-sonnet-4-6`). NQP puede usar cualquier modelo para asistencia de investigación.

---

## Deuda técnica

Ver `docs/` cuando se abra el primer DT.

---

## Licencia

CC BY-NC 4.0 (docs) + AGPL-3.0 (src) — igual que CAL.
