# NQP — Formalización del Operador de Preparación de Estado

**Estado:** Borrador inicial · 2026-06-24
**Autores:** Juan Pablo Chancay, Claude Sonnet 4.6
**Proyecto:** Natural Quantization via State Preparation (NQP)

---

## 1. Motivación

La cuantización estándar de LLMs impone una rejilla discreta uniforme sobre un espacio de pesos
continuo. Esto introduce ruido de cuantización porque la rejilla no respeta la geometría real
de la distribución de pesos: algunos pesos críticos colapsan a bins incorrectos, mientras que
bins de alta resolución se desperdician en regiones de baja varianza.

**Intuición central (J.P. Chancay, 2026-06-24):** igual que en mecánica cuántica un sistema
en superposición colapsa a un eigenvalor *del operador de medición*, los pesos de un modelo
deberían colapsar a bins discretos *del operador natural del modelo* — no de una rejilla
externa. La cuantización óptima es aquella en que la rejilla emerge de la geometría del modelo,
no se le impone.

---

## 2. Definiciones base

Sea $W \in \mathbb{R}^{d}$ el vector de pesos de una capa (o grupo de capas) de un LLM.

**Cuantización estándar** aplica:

$$Q_{\text{std}}(W) = s \cdot \text{round}\!\left(\frac{W}{s}\right) + z$$

con $s$ (scale) y $z$ (zero-point) fijos por rango estadístico de $W$. El error es:

$$\varepsilon_{\text{std}} = \|W - Q_{\text{std}}(W)\|^2$$

**Objetivo NQP:** encontrar una transformación $T: \mathbb{R}^d \to \mathbb{R}^d$ tal que:

$$\varepsilon_{\text{NQP}} = \|W - T^{-1}(Q_{\text{std}}(T(W)))\|^2 \ll \varepsilon_{\text{std}}$$

con $T$ invertible, eficientemente computable, y que preserve la capacidad inferencial del modelo.

---

## 3. El operador de preparación $\hat{P}$

**Definición 3.1** — *Operador de preparación natural*

Sea $\mathcal{H}_W$ el espacio de pesos con producto interno definido por la métrica de Fisher
del modelo:

$$\langle u, v \rangle_F = \mathbb{E}_{x \sim \mathcal{D}}\left[u^T \nabla^2_W \mathcal{L}(x; W) \, v\right]$$

donde $\mathcal{D}$ es la distribución de calibración y $\mathcal{L}$ es la loss del modelo.

El operador de preparación $\hat{P}$ es la transformación lineal ortogonal (o cuasi-ortogonal)
que diagonaliza la métrica de Fisher local:

$$\hat{P} = U \quad \text{tal que} \quad U^T F U = \Lambda$$

donde $F$ es la matriz de Fisher (o su aproximación de bloque) y $\Lambda$ es diagonal.

**Interpretación:** en la base de $\hat{P}$, las direcciones del espacio de pesos son
*independientes bajo la loss* — análogo a la base de eigenvectores del Hamiltoniano en QM,
donde cada dirección tiene energía (impacto) definido.

---

## 4. Cuantización en la base natural

Tras aplicar $\hat{P}$:

$$\tilde{W} = \hat{P} W$$

cada componente $\tilde{W}_i$ tiene varianza proporcional a $\lambda_i^{-1}$ (inversa del
eigenvalor de Fisher correspondiente). Esto permite:

1. **Asignación de bits adaptativa:** componentes con $\lambda_i$ grande (alta curvatura de
   loss → alta sensibilidad) reciben más bits; componentes con $\lambda_i$ pequeño reciben menos.

2. **Rejilla no uniforme:** los bins discretos se distribuyen según la varianza de $\tilde{W}_i$,
   no uniformemente.

3. **Cuantización:** $\hat{W}_i = Q_i(\tilde{W}_i)$ con $Q_i$ específico por componente.

4. **Reconstrucción:** $\hat{W} = \hat{P}^{-1} \hat{\tilde{W}}$

---

## 5. Análogo cuántico explícito

| Mecánica cuántica | NQP |
|---|---|
| Estado pre-medición $\|\psi\rangle$ | Pesos FP32 $W$ |
| Superposición en base computacional | Distribución continua en $\mathbb{R}^d$ |
| Hamiltoniano $\hat{H}$ | Métrica de Fisher $F$ |
| Cambio a base de eigenvectores de $\hat{H}$ | Transformación $\hat{P} = U$ (diagonalización de $F$) |
| Eigenvalores de energía $E_n$ | Eigenvalores de Fisher $\lambda_i$ (curvatura de loss) |
| Colapso al eigenvalor más cercano | Cuantización $Q_i$ en la base natural |
| "Campo EM" que prepara el estado | Fine-tuning / calibración que ajusta $W$ hacia los bins |
| Error de medición mínimo en base propia | Error de cuantización mínimo en base de Fisher |

---

## 6. Propiedad objetivo (conjetura NQP-C1)

**Conjetura:** para cualquier modelo $M$ con pesos $W$ y cualquier presupuesto de bits $b$,
existe un operador de preparación $\hat{P}$ tal que:

$$\text{PPL}(M_{\hat{P},b}) \leq \text{PPL}(M_{\text{std},b}) + \delta$$

con $\delta \to 0$ cuando el número de muestras de calibración $n \to \infty$, donde
$\text{PPL}$ es la perplejidad del modelo cuantizado sobre una distribución de evaluación
y $M_{\hat{P},b}$ es el modelo cuantizado via NQP con $b$ bits.

**Forma más fuerte (NQP-C2):** $\hat{P}$ óptimo lleva $\delta < 0$ — es decir, la cuantización
en base natural con $b$ bits supera a FP32 sin cuantización en tareas donde la distribución
de calibración es representativa, porque $\hat{P}$ actúa como regularizador natural.

---

## 7. Conexión con trabajo existente

| Método | Relación con NQP |
|---|---|
| GPTQ | Minimiza error de cuantización por capa usando Hessiana de outputs — aproximación de bloque de $F$ |
| QuIP / QuIP# | Aplica rotación ortogonal aleatoria — caso especial de $\hat{P}$ sin estructura de Fisher |
| AWQ | Pondera el error por activaciones — aproximación diagonal de $\hat{P}$ |
| SmoothQuant | Rebalanceo por canal — caso 1D de la transformación |
| **NQP** | Generalización: $\hat{P}$ óptimo dado $F$, con asignación de bits adaptativa por eigenvalor |

La novedad de NQP sobre QuIP es que la rotación **no es aleatoria** — es la base de Fisher
del modelo, que tiene significado en términos de sensibilidad de la loss.

---

## 8. Preguntas abiertas

1. ¿Es $F$ computable eficientemente para modelos de escala LLM (7B–70B)?
   → Aproximaciones: K-FAC, diagonal, bloque por capa.

2. ¿La conjetura NQP-C2 (forma fuerte) se cumple en la práctica?
   → Experimento: comparar PPL de NQP vs GPTQ vs QuIP en Llama-3 8B a 4 bits.

3. ¿Cuál es la complejidad de computar $\hat{P}$ vs el ahorro en calidad de inferencia?
   → Trade-off de overhead de preparación vs ganancia de calidad.

4. ¿Existe una noción de "principio de incertidumbre" en NQP?
   → Si $\hat{P}$ diagonaliza $F$, ¿hay direcciones donde precisión de pesos y
   precisión de activaciones no pueden optimizarse simultáneamente?

---

## 9. Próximos pasos

- [ ] Implementar estimación de Fisher diagonal para un transformer pequeño (GPT-2)
- [ ] Comparar $\varepsilon_{\text{NQP}}$ vs $\varepsilon_{\text{std}}$ en distribución controlada
- [ ] Verificar conjetura NQP-C1 empíricamente
- [ ] Formalizar NQP-C2 como teorema con condiciones suficientes
