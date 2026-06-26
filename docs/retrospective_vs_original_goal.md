# NQP — Qué significan los resultados frente al objetivo inicial

**Fecha:** 2026-06-25 · **Investigador:** Juan Pablo Chancay

Este documento cierra el arco del proyecto: contrasta lo obtenido con la **hipótesis fundacional**
declarada en `CLAUDE.md`, sin maquillar el desenlace.

---

## 1. El objetivo inicial (recordatorio textual)

> **NQP — Natural Quantization via State Preparation.** Optimizar la cuantización de LLMs mediante
> transformaciones que alineen la rejilla discreta con la geometría natural del espacio de pesos.
>
> **Intuición:** en mecánica cuántica un sistema colapsa a un eigenvalor del operador de medición;
> la discreción *emerge* de la geometría. En cuantización ML estándar la rejilla se *impone*
> externamente y genera ruido porque no respeta esa geometría.
>
> **Hipótesis NQP:** existe un operador de preparación P̂ (de la métrica de Fisher) tal que
> cuantizar en la base de P̂ minimiza el error — análogo a medir en la base propia del Hamiltoniano.

En una frase: **¿se puede cuantizar mejor rotando primero a una "base natural" derivada de Fisher?**

---

## 2. La respuesta directa a esa pregunta: **NO**

La hipótesis central (NQP-C1) fue **refutada empíricamente**. En orden:

- **Fisher diagonal (EXP-001):** la base de P̂ colapsa a la identidad (P̂ = I) — no hay rotación,
  no hay "base natural" distinta de la canónica. La cuantización en esa base es idéntica a RTN.
- **Fisher por bloques con rotación real (A-G4):** una rotación genuina P̂ ≠ I **no supera** a las
  líneas base fuertes (GPTQ + AWQ + QuIP). La analogía con "medir en la base del Hamiltoniano"
  resultó **metafórica**: el Fisher de activaciones es de rango ~2, no el operador rico que la
  intuición cuántica presuponía.

**Conclusión sobre el objetivo declarado:** NQP, *como método de cuantización*, no existe como
ventaja. La geometría de Fisher de las activaciones no es el "Hamiltoniano" que justificaría una
base de medición privilegiada. El proyecto **no produjo** la herramienta de deployment que se
imaginó al inicio.

---

## 3. Por qué el proyecto no terminó ahí: el pivote

La refutación dejó una pregunta más fina en pie. Si la analogía cuántica falla en *cuantización*,
¿falla por completo, o hay una parte de la estructura cuántica que **sí** describe al transformer?
Eso reorientó el proyecto en dos saltos:

1. **Principio de incertidumbre peso/activación (NQP-U).** ¿Las bases de Fisher de pesos y de
   activaciones *no conmutan* (como observables incompatibles)? **U1a: sí** (ángulo 48.8° vs 83°
   aleatorio — incompatibilidad real). **U1b: no hay consecuencia operativa** (la correlación
   ángulo↔error es espuria: cae a −0.04 al controlar por ε_W). Verdad geométrica, sin payoff en
   cuantización.

2. **Caracterización termodinámica/geométrica de la atención** (línea sucesora). Aquí el ancla
   dejó de ser metáfora: `softmax(QKᵀ/√d)` **es** literalmente una distribución de Boltzmann.
   Esa identidad exacta abrió la línea que sí produjo resultados positivos.

---

## 4. Lo que el proyecto SÍ descubrió (y que no buscaba)

El resultado central del proyecto **no es de cuantización** sino **de geometría de
representaciones**, recogido en el paper *"A Scale-Invariant Atlas of Head-Specific Manifolds in
Transformer Residual Attention"*:

| Hallazgo | Estado |
|---|---|
| Residual de atención ε = Attn − V_{i\*} vive en **manifold no-lineal ~7D** por cabeza (vs ~30 lineal) | ✅ robusto |
| Las cabezas ocupan **subespacios mutuamente no-alineados** (overlap O_h ≈ 0.28) | ✅ robusto, **resultado central** |
| La no-alineación es **invariante a escala** (small/medium/large, CIs solapan) y **a corpus** (WikiText vs C4) | ✅ robusto |
| Estructura de fases termodinámica (líquido→cristal) + S_vn como **proxy de incertidumbre** | ✅ resultado bridge |
| No comprimible por selección dura (Top-k), por low-rank lineal, ni por autoencoder no-lineal por-cabeza | ✅ tres negativos limpios |

La "geometría natural del espacio" que NQP buscaba **existe** — pero no en los *pesos* (donde se la
buscó para cuantizar) sino en el *residual de la atención*, y **no se traduce en compresión**.

---

## 5. La lección epistemológica (el verdadero retorno del proyecto)

El objetivo inicial era una **apuesta sobre una analogía física**. El valor del proyecto no fue
confirmarla sino **medir con disciplina dónde la analogía es literal y dónde es decorativa**:

- **Literal y útil:** softmax = Boltzmann → termodinámica de la atención medible.
- **Real pero inerte:** no-conmutatividad peso/activación (incertidumbre sin consecuencia).
- **Decorativa:** "base del Hamiltoniano" para cuantizar (Fisher de activaciones rango ~2).

Patrón metodológico recurrente que el proyecto consolidó: **la herramienta ingenua sobre-concluye;
el control correcto reenmarca.** Apareció cuatro veces — error L2 sesga contra NQP; correlación
bivariada vs parcial en U1b; rango lineal (PCA) vs dimensión intrínseca (TwoNN); y "atlas" como
fiber-bundle formal vs descriptivo. Cada vez, el control honesto evitó una sobre-afirmación.

---

## 6. Veredicto frente al objetivo inicial

- **¿Se cumplió el objetivo declarado (cuantización natural vía Fisher)?** **No.** Refutado y
  documentado, no abandonado en silencio.
- **¿El proyecto fracasó?** **No.** Convirtió una hipótesis física refutada en un resultado de
  interpretabilidad positivo, reproducible, delimitado y con controles negativos — un paper.
- **¿Sigue viva la relación con CAL/L2 (deployment) que CLAUDE.md anticipaba?** No por la vía
  prevista: NQP no es infraestructura de cuantización. Su contribución es conceptual
  (representación geométrica de la atención), no de deployment.

> NQP empezó preguntando *cómo medir mejor para discretizar pesos* y terminó respondiendo *cómo
> está organizada geométricamente la atención*. El objetivo original quedó refutado; el método de
> trabajo que se usó para refutarlo produjo el resultado que vale.
