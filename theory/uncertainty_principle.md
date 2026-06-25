# NQP — Principio de Incertidumbre Peso/Activación

**Estado:** Borrador inicial · 2026-06-24
**Autores:** Juan Pablo Chancay, Claude Sonnet 4.6
**Proyecto:** Natural Quantization via State Preparation (NQP)

---

## 0. Por qué este pivote

Los experimentos EXP-001 y A-G4 (ver [ROADMAP](../experiments/ROADMAP.md)) demostraron que la
parte de **cuantización** de NQP no tiene contenido cuántico genuino: la base de Fisher de
**activaciones** colapsa a la combinación GPTQ (segundo orden) + AWQ (outliers) + QuIP
(incoherencia), y la analogía con "medir en la base del Hamiltoniano" resultó metafórica
— la Fisher empírica de activaciones es rango ~2 (PR ≈ 1.5–2.0 de 768), sin el espectro
rico que la analogía requiere.

**La parte genuinamente cuántica de NQP es otra:** la conjetura de que existe un
**principio de incertidumbre** entre la precisión de pesos y la de activaciones. Esto NO lo
aborda ningún método de cuantización existente, y aquí la analogía cuántica deja de ser
metáfora — se vuelve estructura matemática literal (operadores que no conmutan).

---

## 1. Las dos métricas de Fisher

NQP define dos objetos geométricos distintos sobre la misma capa:

**Fisher respecto a pesos** $F_W$ — cómo cambia la loss al perturbar los **pesos** $W$:
$$F_W = \mathbb{E}_{x}\left[ \nabla_W \mathcal{L}(x)\, \nabla_W \mathcal{L}(x)^T \right]$$
Captura qué direcciones del *espacio de pesos* importan para la loss. Es lo que un
cuantizador de pesos querría diagonalizar.

**Fisher respecto a activaciones** $G_A$ — cómo cambia la loss al perturbar las
**activaciones** $a$ (entradas a la capa):
$$G_A = \mathbb{E}_{x}\left[ a\, a^T \right] \quad (\text{Gauss-Newton}) \;\;\text{o}\;\;
\mathbb{E}_{x}\left[ \nabla_a \mathcal{L}\, \nabla_a \mathcal{L}^T \right]$$
Captura qué direcciones del *espacio de activaciones* importan. Es lo que GPTQ/AWQ usan.

**Observación clave:** estos viven en el mismo $\mathbb{R}^{d_{in}}$ (vía la estructura
lineal $y = Wa$), así que sus bases propias son comparables — y la pregunta de si conmutan
está bien definida.

---

## 2. El conmutador

Sean $\hat{P}_W$ la base que diagonaliza $F_W$ y $\hat{P}_A$ la que diagonaliza $G_A$.

**Definición 2.1** — *Conmutador de preparación*
$$[\hat{P}_W, \hat{P}_A] := \hat{P}_W \hat{P}_A - \hat{P}_A \hat{P}_W$$

**Conjetura NQP-U1 (principio de incertidumbre):**
$$[\hat{P}_W, \hat{P}_A] \neq 0 \;\;\Longrightarrow\;\; \exists\, c > 0 :\;
\varepsilon_W \cdot \varepsilon_A \;\geq\; c$$
donde $\varepsilon_W$ es el error mínimo de cuantización de pesos y $\varepsilon_A$ el de
activaciones, alcanzables simultáneamente. Es decir: **no se puede minimizar ambos a la vez**
si las bases no conmutan — exactamente como $\Delta x \, \Delta p \geq \hbar/2$.

**Interpretación:** $F_W$ y $G_A$ juegan el papel de dos observables incompatibles. La base
que es óptima para cuantizar pesos NO es óptima para cuantizar activaciones, y el grado de
incompatibilidad lo cuantifica $\|[\hat{P}_W,\hat{P}_A]\|$.

---

## 3. El análogo cuántico (ahora literal, no metafórico)

| Mecánica cuántica | NQP-U |
|---|---|
| Observable posición $\hat{x}$ | Fisher de pesos $F_W$ |
| Observable momento $\hat{p}$ | Fisher de activaciones $G_A$ |
| $[\hat{x},\hat{p}] = i\hbar \neq 0$ | $[\hat{P}_W,\hat{P}_A] \neq 0$ |
| $\Delta x \, \Delta p \geq \hbar/2$ | $\varepsilon_W \, \varepsilon_A \geq c$ |
| Bases propias incompatibles | Bases de cuantización incompatibles |
| $\hbar$ (cuanto de acción) | $c$ (cuanto de error conjunto) |

A diferencia de la cuantización (donde la analogía era decorativa), aquí la no-conmutatividad
es una **propiedad medible** de la red, y la cota de incertidumbre es **falsable**.

---

## 4. Predicciones falsables

**NQP-U1a:** En capas reales de un transformer, $\|[\hat{P}_W,\hat{P}_A]\| > 0$ de forma
estadísticamente significativa (las bases NO conmutan). → *Test directo: medir el conmutador.*

**NQP-U1b:** El producto $\varepsilon_W \cdot \varepsilon_A$ está acotado inferiormente y la
cota correlaciona con $\|[\hat{P}_W,\hat{P}_A]\|$. → *Test: barrer asignaciones de bits entre
pesos y activaciones, trazar la frontera de Pareto, ver si hay un "suelo".*

**NQP-U1c (fuerte):** La cota $c$ predice el degradación mínima alcanzable por CUALQUIER
esquema de cuantización conjunta peso+activación. Si es cierto, NQP-U da un **límite
fundamental** — algo que ni GPTQ ni AWQ ni QuIP proveen (todos optimizan un lado a la vez).

---

## 5. Por qué esto SÍ es novel

- GPTQ/AWQ/QuIP cuantizan **pesos** usando geometría de **activaciones**. Ninguno modela la
  tensión peso↔activación como un trade-off fundamental.
- La cuantización de activaciones (p.ej. SmoothQuant) y la de pesos se tratan como problemas
  separados o se balancean heurísticamente. NQP-U propone que hay un **límite teórico** a ese
  balance, derivado de la no-conmutatividad.
- Si NQP-U1c se sostiene, el aporte no es "otro método de cuantización" (mercado saturado)
  sino una **cota de imposibilidad** — un resultado de tipo distinto, más cercano a teoría
  de la información que a ingeniería de kernels.

---

## 6. Resultado EXP-U01 (2026-06-24) — NQP-U1a SOPORTADA

Medición de las bases propias de $F_W$ (Fisher de pesos) y $G_A$ (Fisher de activaciones)
en GPT-2 small, 64 muestras de calibración WikiText-103, subespacio dominante top-16:

| Referencia | Ángulo principal (top-16) | top-eigvec overlap |
|---|---|---|
| Bases conmutan (misma base) | 0° | 1.000 |
| **GPT-2 observado** | **48.8°** | **0.434** |
| Bases aleatorias (control, 5 seeds) | 83.0° ± 0.1 | 0.030 ± 0.02 |

**Lectura:** las bases de Fisher-peso y Fisher-activación de GPT-2 caen en un punto
**intermedio genuino** entre conmutar (0°) y ser aleatorias (83°):
- **No conmutan** (48.8° >> 0°) → existe conflicto entre la base óptima para pesos y la
  óptima para activaciones. *Precondición del principio de incertidumbre satisfecha.*
- **No son aleatorias** (48.8° << 83°; overlap 0.434 vs 0.030 = 14× sobre el azar) → el
  conflicto es estructural, no trivial. Si fueran aleatorias, cualquier par de bases lo
  mostraría y el resultado no significaría nada.
- **Varía por tipo de capa:** `lm_head` da 21.7° (casi alineada), atención ~45-48°. La
  estructura depende de la función de la capa → señal real, no ruido uniforme.

**Caveat metodológico:** la norma del conmutador full-rank $\|[\hat{P}_W,\hat{P}_A]\|_F$
**satura a 1.0 en alta dimensión** (cualquier par no idéntico en d=768 la satura, incl.
aleatorio). NO es la métrica decisiva. La métrica válida son los **ángulos principales del
subespacio dominante** comparados contra el control aleatorio.

## 7. Próximos pasos

- [x] EXP-U01: medir alineación de bases $F_W$ / $G_A$ (NQP-U1a) → **SOPORTADA (48.8°)**.
- [ ] EXP-U02: frontera de Pareto $\varepsilon_W$ vs $\varepsilon_A$ (NQP-U1b) — ¿hay un
      "suelo" en el producto $\varepsilon_W \cdot \varepsilon_A$, y correlaciona con el
      ángulo por capa?
- [ ] Formalizar la cota $c$ en función de los espectros de $F_W$, $G_A$ y su ángulo.
- [ ] Robustez: repetir con más muestras (64→256) y verificar estabilidad del ángulo.
