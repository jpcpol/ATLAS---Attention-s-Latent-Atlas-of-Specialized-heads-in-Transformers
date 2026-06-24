# NQP — Preguntas de investigación abiertas

**Última actualización:** 2026-06-24

---

## RQ-1 (Central)
¿Existe un operador de preparación $\hat{P}$ derivado de la métrica de Fisher de un LLM
tal que cuantizar en la base de $\hat{P}$ minimice el error de cuantización respecto a
cualquier método de cuantización con rejilla fija?

**Hipótesis:** Sí. La base de Fisher es la base "natural" del modelo en el sentido de que
en ella las direcciones de pesos son independientes bajo la loss — y cuantizar direcciones
independientes con presupuesto de bits proporcional a su curvatura es óptimo (argumento
de rate-distortion).

---

## RQ-2
¿Es la matriz de Fisher $F$ computable de forma suficientemente eficiente para modelos
de escala práctica (7B–70B parámetros)?

**Estado del arte:** K-FAC (Kronecker-factored approximation), diagonal Fisher, Fisher de
bloque por capa. La pregunta es si alguna aproximación preserva suficiente estructura para
que $\hat{P}$ derivado de ella sea mejor que una rotación aleatoria (QuIP).

---

## RQ-3
¿Existe una noción de "principio de incertidumbre" en el espacio de pesos?

**Intuición:** si $\hat{P}$ diagonaliza $F$, puede haber direcciones donde precisión de
pesos y precisión de activaciones no pueden optimizarse simultáneamente — análogo al
principio de incertidumbre de Heisenberg entre posición y momento.

**Formalización tentativa:** para $\hat{P}$ que diagonaliza $F_W$ (Fisher respecto a pesos)
y $G_A$ (Fisher respecto a activaciones), si $[\hat{P}_W, \hat{P}_A] \neq 0$, entonces
existe un trade-off fundamental entre error de cuantización en pesos y error de activaciones.

---

## RQ-4
¿NQP-C2 (forma fuerte de la conjetura) puede ser verdadera?

La conjetura dice que cuantizar en base natural con $b$ bits puede *superar* FP32. Esto
implicaría que $\hat{P}$ actúa como regularizador: al eliminar las componentes de baja
curvatura (poco relevantes para la loss), se reduce overfitting a ruido de entrenamiento.

**Análogía:** la cuantización en base natural sería equivalente a truncar los eigenvalores
de Fisher pequeños — similar a low-rank approximation o dropout estructurado.

---

## RQ-5
¿Qué relación tiene NQP con los métodos existentes?

- GPTQ usa la Hessiana de outputs por capa → aproximación de $F$ local
- QuIP usa rotaciones ortogonales aleatorias → $\hat{P}$ sin estructura
- AWQ usa escala por canal → $\hat{P}$ diagonal
- **NQP usa $F$ global (o por bloque) con estructura de eigenvalores** → generalización

Si NQP = QuIP cuando $F = I$ (Fisher isotrópico), la conexión es exacta y NQP es
estrictamente más general.

---

## RQ-6 (Aplicada)
¿Puede NQP usarse como capa de infraestructura para modelos dentro de CAL?

Si NQP produce modelos cuantizados con menor degradación, el evaluador LLM de CAL/L2
(actualmente `claude-sonnet-4-6` vía API) podría eventualmente reemplazarse por un modelo
local cuantizado via NQP con calidad equivalente — reduciendo costos de inferencia del
experimento y eliminando la dependencia de API externa.

**Condición:** el modelo NQP-cuantizado debe pasar el gate φ de CAL (ρ ≥ 0.75 en el
benchmark DT-021).
