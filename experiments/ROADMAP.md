# NQP — Roadmap de comprobaciones A → B → C

**Última actualización:** 2026-06-24
**Estado de partida:** EXP-001 (Fisher diagonal, $\hat{P}=I$) implementado y validado.

**Resultado del barrido de calibración (2026-06-24):**

| n_calib | bits | PPL std | PPL NQP | Δ | NQP gana (err L2) |
|---|---|---|---|---|---|
| 32 | 8 | 34.06 | 36.67 | +2.61 | 0/48 |
| 64 | 8 | 34.06 | 36.62 | +2.55 | 0/48 |
| 128 | 8 | 34.06 | 37.02 | +2.96 | 0/48 |
| 128 | 4 | 9808 | 7175 | (ambas basura) | **20/48** |

**Lectura:**
- **8 bits → diagonal plano y muerto.** Δ no se mueve con calibración (32→128). El problema
  es estructural ($\hat{P}=I$), no de muestras. **Caso cerrado.**
- **4 bits → aparece palanca cruda.** 20/48 capas reducen error L2 (vs 0/48 a 8 bits), pero
  ambas PPL son basura (RTN a 4-bit destruye GPT-2) y el diagonal no puede convertir la
  reducción de error-L2 en PPL porque le falta la rotación.
- **Firma clara de que falta $\hat{P}\neq I$.** La señal a 4 bits es exactamente lo que el
  Camino A debería poder explotar — y el baseline correcto a 4 bits es GPTQ, no RTN.

**Gate de entrada a Camino A: PASADO.**

---

## Por qué A → B → C (y no otro orden)

La intuición central de NQP (CLAUDE.md §"Intuición fundacional") es *cuantizar en la
**base propia** del operador natural del modelo* — análogo a medir en la base del
Hamiltoniano. Esa base propia es $\hat{P}=U$ con $U^T F U = \Lambda$
([operator_formalization.md:58](../theory/operator_formalization.md)).

**El diagonal probado en EXP-001 tiene $U=I$: no hay base propia, no hay rotación.**
Se reduce a escalado adaptativo por elemento (≈ AWQ). Por eso pierde. La hipótesis NQP-C1
**nunca fue testeada de verdad** — testeamos su sombra trivial.

Los tres caminos son **niveles crecientes de fidelidad a la intuición**, ordenados por
relación señal/esfuerzo:

| | Camino | $\hat{P}$ | Encarna la intuición | Esfuerzo | Función en el roadmap |
|---|---|---|---|---|---|
| **A** | Fisher por-bloque + rotación real | $U \neq I$ (eig de bloque) | ✅ Literal | Medio | **Prueba de existencia** de NQP-C1 |
| **B** | K-FAC (Kronecker-factored) | $U_A \otimes U_G$ | ✅ Escalable | Alto | **Escala** A a LLMs reales |
| **C** | Rate-distortion sobre eigenespectro | $U$ + asignación óptima de bits | ✅✅ Forma fuerte | Alto | **Teorema** (NQP-C1/C2 formal) |

A es la apuesta de menor riesgo que toca la idea real. B solo tiene sentido si A funciona.
C es el paper, y solo se escribe si A+B dan señal empírica.

---

## CAMINO A — Fisher por-bloque con rotación real

**Objetivo:** instanciar $\hat{P}=U \neq I$ y verificar si cuantizar en la base de Fisher
(no aleatoria, a diferencia de QuIP) bate al baseline a bits bajos.

**Definición operativa:**
- Para cada matriz de pesos $W \in \mathbb{R}^{d_{out}\times d_{in}}$, estimar el Fisher
  de bloque sobre las columnas de entrada: $F \in \mathbb{R}^{d_{in}\times d_{in}}$
  (Gauss-Newton / empirical Fisher de activaciones de entrada — igual estructura que la
  Hessiana de GPTQ).
- Diagonalizar: $F = U\Lambda U^T$.
- Rotar pesos a la base propia: $\tilde{W} = W U$.
- Cuantizar $\tilde{W}$ con escala por-columna derivada de $\lambda_i$.
- Reconstruir: $\hat{W} = Q(\tilde{W})\,U^T$.

**Comprobaciones (gates):**
- **A-G1** — sanity: $U^TU = I$ (ortogonalidad numérica), reconstrucción FP32 sin cuantizar
  recupera $W$ con error < 1e-5.
- **A-G2** — error L2: $\varepsilon_{NQP} < \varepsilon_{std}$ en ≥ 60% de las capas a **4 bits**
  (a 8 bits hay poca señal; el régimen interesante es 4/3 bits).
- **A-G3** — PPL: $\text{PPL}_{NQP} \leq \text{PPL}_{GPTQ}$ en GPT-2 a 4 bits (GPTQ es el
  comparador justo, no INT4-RTN).
- **A-G4** — ablación clave: ¿la rotación de **Fisher** bate a una rotación **aleatoria**
  (QuIP) con el mismo presupuesto? Si NO, la estructura de Fisher no aporta y NQP colapsa
  a QuIP. **Este es el gate que decide si NQP existe como método.**

**Salida:** si A-G4 pasa → NQP tiene contenido empírico → proceder a B.
Si A-G4 falla → la intuición es elegante pero la base de Fisher no supera al azar; pivotar
a investigar *por qué* (¿Fisher mal estimado? ¿bloque demasiado pequeño?) antes de abandonar.

**Costo estimado:** ~3–4 sesiones. Diagonalización de bloques 768×768 es trivial en CPU.

---

## CAMINO B — K-FAC (Kronecker-factored)

**Precondición:** A-G4 pasó (la base de Fisher aporta sobre azar).

**Objetivo:** hacer A escalable. El Fisher de bloque completo es $O(d^2)$ por capa;
inviable a 7B+. K-FAC factoriza $F \approx A \otimes G$ (input-cov ⊗ output-grad-cov),
reduciendo a dos eig pequeñas y $\hat{P} = U_A \otimes U_G$.

**Comprobaciones (gates):**
- **B-G1** — fidelidad: la base K-FAC reproduce ≥ 90% de la ganancia de A-G3 en GPT-2
  (verificar que la aproximación Kronecker no destruye la señal).
- **B-G2** — escala: correr en Llama-3 8B a 4 bits, $\text{PPL}_{NQP} \leq \text{PPL}_{GPTQ}$
  (= EXP-002 del README).
- **B-G3** — overhead: tiempo de preparar $\hat{P}$ < 2× el de GPTQ (RQ-3: trade-off
  overhead vs calidad).

**Salida:** si B pasa → NQP es un método de cuantización viable y competitivo → C.

**Costo estimado:** ~6–8 sesiones (implementación K-FAC + acceso a GPU para Llama-8B).

---

## CAMINO C — Rate-distortion sobre el eigenespectro

**Precondición:** A+B dan señal empírica reproducible.

**Objetivo:** la forma fuerte. No solo rotar, sino **asignar bits óptimamente** por
eigenvalor vía rate-distortion: $b_i \propto \log \lambda_i$ (más bits donde la loss es
más curva), y formalizar NQP-C1 como teorema con condiciones suficientes
([operator_formalization.md:104](../theory/operator_formalization.md)).

**Comprobaciones (gates):**
- **C-G1** — teoría: derivar la asignación de bits óptima bajo presupuesto fijo y demostrar
  cota $\varepsilon_{NQP} \leq \varepsilon_{std}$ bajo Fisher exacto.
- **C-G2** — NQP-C2 (forma fuerte): test empírico de si NQP-4bit **supera FP32** en tareas
  donde la calibración es representativa (= EXP-003, hipótesis del regularizador).
- **C-G3** — paper: resultados reproducibles + comparación vs GPTQ/QuIP#/AWQ.

**Costo estimado:** indefinido — es el cuerpo del paper.

---

## Decisión de "cuál camino explotar"

**Veredicto de diseño:** explotar **A primero, en profundidad**, porque:

1. Es el **único** que puede *falsar* NQP barato. A-G4 (Fisher vs rotación aleatoria) es la
   pregunta científica central — si la base de Fisher no bate al azar, todo lo demás es
   decoración. Ningún otro camino responde esto más rápido.
2. B y C **heredan** toda su validez de A. Invertir en K-FAC o en la teoría rate-distortion
   antes de saber si la rotación de Fisher aporta es construir sobre arena.
3. A reutiliza ~70% del código actual (`fisher.py`): solo añade el bloque de
   diagonalización + rotación. El diagonal ya hecho queda como ablación ($U=I$).

**Anti-patrón a evitar:** saltar a B (K-FAC, "lo escalable") porque suena más impresionante.
Si A no pasa A-G4 en GPT-2 (124M, minutos de cómputo), B en Llama-8B (horas de GPU) solo
desperdiciará recursos confirmando lo mismo a mayor costo.

---

## Próxima acción concreta

Implementar `src/fisher_block.py` (Camino A) con:
- `estimate_block_fisher(model, calib)` → $F$ por matriz de pesos.
- `NQPBlockQuantizer` con rotación $\tilde W = WU$ / reconstrucción $\hat W = Q(\tilde W)U^T$.
- Comparador A-G4: NQP-Fisher vs NQP-random-rotation vs GPTQ-baseline.
- Gate de entrada: esperar veredicto del barrido de calibración (EXP-001) para confirmar
  que el diagonal es plano antes de invertir en A.
