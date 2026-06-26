# ATLAS — *Attention's Latent Atlas of Specialized heads*

> Formerly **NQP** (*Natural Quantization via State Preparation*). The founding quantization
> hypothesis was refuted; the project's destination turned out to be **the geometry it uncovered
> on the way**: a scale-invariant atlas of head-specific, mutually non-aligned manifolds in the
> residual of transformer attention. The repository keeps the name **NQP** on disk for git
> continuity, but the project's *goal* — and this README — point at the atlas.

**Investigador principal:** Juan Pablo Chancay · jpcpol@gmail.com
**Inicio:** 2026-06-24 · **Venue objetivo:** NeurIPS / ICLR (workshop)
**Licencia:** ver [LICENSE.md](LICENSE.md) — CC BY-NC 4.0 (docs/theory) + AGPL-3.0 (src)

---

## El arco en una frase

El proyecto empezó preguntando *"¿podemos cuantizar mejor un LLM rotando los pesos a la base
natural de Fisher (análogo a medir en la base del Hamiltoniano)?"* — y terminó respondiendo *"esa
idea no funcionó, pero refutarla reveló una estructura geométrica nueva y reproducible en la
atención: cada cabeza vive en un manifold ~7D propio, y esos manifolds están mutuamente
no-alineados (overlap O_h ≈ 0.28), de forma estable a escala y corpus."*

| Hipótesis original (refutada) | Resultado superviviente (positivo) |
|---|---|
| Cuantización en la base de Fisher P̂ bate a GPTQ+AWQ+QuIP | Fisher de activaciones es rango ~2 → colapsa a las líneas base. **Refutada.** |
| Existe un método de *deployment* mejor | Lo que hay es un objeto a nivel de **representación**, no un truco de compresión |
| La cola de atención es comprimible | Atlas escala-invariante de manifolds por cabeza (O_h ≈ 0.28), geométricamente real pero **no funcionalmente comprimible** por autoencoder simple (Q06) |

Documento que cierra este arco: [docs/retrospective_vs_original_goal.md](docs/retrospective_vs_original_goal.md)
y la §0 del paper ([docs/paper_draft.md](docs/paper_draft.md)).

---

## Objetivos (estrella polar)

- **Objetivo científico (en gran parte cumplido):** comprender la organización geométrica de la
  atención y su papel funcional. → *Existe un atlas de manifolds por cabeza, no alineados, robusto
  a escala y corpus.*
- **Objetivo aplicado (abierto):** determinar si dicha organización puede explotarse para construir
  Transformers más eficientes, adaptativos o interpretables — **sin repetir el error de NQP**: que
  exista una estructura geométrica no implica que sea explotable.

---

## Estructura del repositorio

```
NQP/
├── README.md          ← este índice
├── LICENSE.md         ← licencia dual (CC BY-NC 4.0 + AGPL-3.0)
├── CLAUDE.md          ← contexto del proyecto para asistencia
├── docs/              ← papers, preguntas, retrospectiva, figuras
├── theory/            ← formalización matemática (operador, mapa cuántico, incertidumbre)
├── experiments/       ← roadmap y especificación de experimentos
└── src/               ← implementación y scripts de medición
```

---

## 📄 `docs/` — Papers, figuras y narrativa

| Archivo | Qué es |
|---|---|
| [docs/paper_draft.md](docs/paper_draft.md) | **Preprint completo (congelado).** Resultado central: atlas escala-invariante (O_h ≈ 0.28). §0 ata los resultados al objetivo original; §3 ordena la evidencia por fuerza; §5 Related Work con referencias [1]–[16]. **Empieza por aquí.** |
| [docs/paper_skeleton.md](docs/paper_skeleton.md) | Esqueleto previo del paper (estructura termodinámica + geometría intrínseca). |
| [docs/research_questions.md](docs/research_questions.md) | Preguntas de investigación abiertas (Q01–Q06 y derivadas). |
| [docs/retrospective_vs_original_goal.md](docs/retrospective_vs_original_goal.md) | Qué significan los resultados frente a la hipótesis fundacional, sin maquillar el desenlace. |
| [docs/figure_data.json](docs/figure_data.json) | Fuente única de datos para las figuras (matrices de overlap, dims, escalares de runs). |
| [docs/figures/](docs/figures/) | Las 8 figuras del paper (5 principales + 3 suplementarias). Fig 1 = matriz de overlap H×H (la icónica). |

---

## 🧮 `theory/` — Formalización matemática

| Archivo | Qué es |
|---|---|
| [theory/operator_formalization.md](theory/operator_formalization.md) | Formalización del operador de preparación P̂ = U (diagonaliza Fisher). Base de la hipótesis original (rama de cuantización). |
| [theory/quantum_transformer_map.md](theory/quantum_transformer_map.md) | Mapa sistemático mecánica cuántica ↔ Transformers. Origen de la línea sucesora; clasifica la analogía en decorativa / inerte / exacta (softmax = Boltzmann). |
| [theory/uncertainty_principle.md](theory/uncertainty_principle.md) | Principio de incertidumbre peso/activación (rama NQP-U). U1a ✅ (bases no conmutan, ángulo ≈49°), U1b ❌ (sin cota operativa). |

---

## 🧪 `experiments/` — Roadmap y especificación

| Archivo | Qué es |
|---|---|
| [experiments/README.md](experiments/README.md) | Experimentos de cuantización (EXP-001…003) — la rama original, cerrada. |
| [experiments/ROADMAP.md](experiments/ROADMAP.md) | Roadmap A→B→C de la cuantización Fisher y el registro de su refutación (gates A-G1…A-G4, error-L2 vs PPL). Lectura clave de **por qué** la idea original no pasó. |

---

## 💻 `src/` — Implementación y medición

### Rama original — cuantización Fisher (resultados negativos documentados)

| Archivo | Función |
|---|---|
| [src/fisher.py](src/fisher.py) | EXP-001: cuantizador Fisher diagonal (P̂ = I). Diagonal plano y muerto. |
| [src/fisher_block.py](src/fisher_block.py) | Camino A: Fisher por-bloque con rotación real (P̂ = U ≠ I). Gate A-G4. |
| [src/uncertainty.py](src/uncertainty.py) | EXP-U01: medir el conmutador [P̂_W, P̂_A] (bases peso/activación). |
| [src/pareto.py](src/pareto.py) | EXP-U02: frontera de Pareto ε_W / ε_A. |

### Rama superviviente — geometría y termodinámica de la atención

| Archivo | Función |
|---|---|
| [src/residual.py](src/residual.py) | Descomposición exacta `Attn = a·V_{i*} + (1−a)·ε`; recolección + SVD del residual; parches Top-1/full/low-rank. |
| [src/intrinsic.py](src/intrinsic.py) | Estimador de dimensión intrínseca **TwoNN** (validado en swiss-roll) + rango lineal PCA(90%). |
| [src/manifold.py](src/manifold.py) | `atlas_test`: conectividad, homogeneidad local, interpolación, overlap de subespacios. |
| [src/thermo.py](src/thermo.py) | Observables de Boltzmann (F, ⟨E⟩, C, T_eff, S_vn) desde la misma Z que la atención computa. |
| [src/crystallize.py](src/crystallize.py) | Baseline Top-k (selección dura) + perplejidad; mide el daño que la geometría debe reparar. |
| [src/scaling.py](src/scaling.py) | Dimensión intrínseca y profundidad de cristalización L_c a través de escala. |
| [src/atlas_scaling.py](src/atlas_scaling.py) | Overlap inter-cabeza O_h a través de la familia GPT-2 (escala-invarianza). |
| [src/atlas_robustness.py](src/atlas_robustness.py) | **Endurecimiento del claim central:** CI bootstrap de O_h + sensibilidad a d_local / N / profundidad. |
| [src/atlas_intercorpus.py](src/atlas_intercorpus.py) | Control inter-corpus (WikiText-103 vs C4): O_h es propiedad del modelo, no del corpus. |
| [src/autoencoder.py](src/autoencoder.py) | **EXP-Q06:** autoencoder no-lineal por cabeza (64→7→64) vs PCA rank-7. Negativo limpio: el manifold no es funcionalmente comprimible por esta vía. |
| [src/figure_data.py](src/figure_data.py) | Recolecta los datos de las figuras → `docs/figure_data.json`. |
| [src/make_figures.py](src/make_figures.py) | Renderiza las 8 figuras → `docs/figures/`. |

---

## Reproducir las figuras

```bash
cd src
python figure_data.py     # → docs/figure_data.json (recolecta matrices fresh, ~minutos en CPU)
python make_figures.py     # → docs/figures/*.png
```

Modelos: familia GPT-2 (124M / 355M / 774M, vía `transformers`). Datos: WikiText-103 validation
(+ C4 para el control inter-corpus). Todas las comparaciones cross-escala fijan N / nº de cabezas /
profundidad relativa.

---

## Estado

| Componente | Estado |
|---|---|
| Cuantización Fisher (NQP-C1) | ❌ Refutada — colapsa a GPTQ+AWQ+QuIP |
| Principio de incertidumbre (NQP-U1) | ⚠️ Parcial — bases no conmutan pero sin consecuencia operativa |
| Atlas escala-invariante (O_h ≈ 0.28) | ✅ Resultado central, congelado |
| Manifold por cabeza ~7D no-lineal | ✅ |
| Compresión funcional del manifold (Q06) | ❌ Negativo honesto (AE ≈ PCA) |
| Preprint | ✅ Borrador completo y congelado |
| Confirmación de metadatos bibliográficos | ⬜ Pendiente (gestor bibliográfico) |
| Cross-arquitectura (Llama / Mistral) | ⬜ Pendiente — promovería "scale-invariant within GPT-2" |

---

## Líneas futuras (priorizadas, con la cautela de NQP)

1. **Routing geométrico entre cabezas** — activación dinámica de un subconjunto de cabezas (espíritu MoE, pero por geometría latente). La apuesta aplicada más prometedora.
2. **Métricas diagnósticas** — usar O_h para detectar colapso/redundancia de cabezas; no requiere cambiar la arquitectura.
3. **Estabilidad del atlas bajo fine-tuning** — ¿el atlas es más estable que los pesos a través de pretraining → instruction tuning → RLHF?
4. **(Especulativo)** compresión estructurada (diccionario compartido + coordenadas por cabeza), regularización geométrica, arquitecturas jerárquicas macro/micro.

La pregunta central de mediano plazo: **¿la geometría es causal o solo descriptiva?**
