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

- **NQP-C1 (cuantización):** ❌ *Refutada empíricamente* (2026-06-24). La base de Fisher de
  activaciones no supera a GPTQ+AWQ+QuIP; la analogía con "medir en la base del Hamiltoniano"
  resultó metafórica (Fisher de activaciones es rango ~2). Ver `experiments/ROADMAP.md`.
- **NQP-U1 (incertidumbre):** 🟢 *Conjetura activa principal.* Existe un principio de
  incertidumbre entre precisión de pesos y activaciones: si $[\hat{P}_W,\hat{P}_A]\neq 0$,
  entonces $\varepsilon_W \cdot \varepsilon_A \geq c$. Esta es la parte con estructura cuántica
  **literal** (observables que no conmutan). Ver `theory/uncertainty_principle.md`.
- **NQP-C2:** congelada (dependía de C1).

---

## Estado

| Componente | Estado |
|---|---|
| Cuantización NQP (C1) | ❌ Refutada — colapsa a GPTQ+AWQ+QuIP |
| Implementaciones EXP-001 / A-G4 | 🟢 `src/fisher.py`, `src/fisher_block.py` (resultados negativos documentados) |
| Formalización principio incertidumbre (U1) | 🟢 `theory/uncertainty_principle.md` |
| EXP-U01 (medir $[\hat{P}_W,\hat{P}_A]$) | 🟡 `src/uncertainty.py` — en ejecución |
| EXP-U02 (frontera de Pareto $\varepsilon_W$/$\varepsilon_A$) | ⬜ Pendiente de U1a |
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
