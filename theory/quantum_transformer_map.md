# Mapa Matemático: Mecánica Cuántica ↔ Transformers

**Estado:** Exploración inicial · 2026-06-25
**Autores:** Juan Pablo Chancay, Claude Opus 4.8
**Origen:** línea sucesora de NQP (en standby). NQP falló por partir de UNA analogía forzada
(cuantización = medición). Aquí invertimos el enfoque: mapear sistemáticamente la maquinaria
matemática compartida, identificar gaps donde MC tiene estructura que los transformers no
explotan, y filtrar por **viabilidad falsable**, no por elegancia.

---

## 0. Disciplina anti-NQP

Cada candidato debe pasar tres filtros antes de invertir cómputo:

1. **Coincidencia literal, no metafórica:** ¿MC y transformers usan *el mismo objeto
   matemático*, o solo "se parecen"? (NQP confundió "Fisher se parece al Hamiltoniano" con
   "Fisher es el Hamiltoniano". Falló.)
2. **Gap real:** ¿la herramienta de MC hace algo que la formulación actual del transformer
   NO hace ya bajo otro nombre? (NQP-C1 colapsó a GPTQ+AWQ+QuIP porque no aportaba nada nuevo.)
3. **Falsable y medible:** ¿hay un experimento barato que pueda *refutar* la utilidad en
   GPT-2 antes de escalar? (NQP-U1b parecía soportada hasta la correlación parcial.)

---

## 1. Coincidencias matemáticas LITERALES

Objetos que ambos campos usan de forma idéntica (no análoga):

| Objeto | En MC | En transformers | ¿Literal? |
|---|---|---|---|
| **Producto interno / proyección** | ⟨ψ\|φ⟩ amplitud | QᵀK score de atención | 🟢 mismo álgebra |
| **Softmax / distribución de Gibbs** | e^{−E/kT}/Z (Boltzmann) | softmax(QKᵀ/√d) | 🟢 misma forma, T↔√d |
| **Normalización L2 / estado** | \|ψ\|=1 (estado normalizado) | LayerNorm, RMSNorm | 🟡 parcial (L2 vs varianza) |
| **Operador lineal / matriz densa** | Â observable hermitiano | W proj (no hermitiana) | 🟡 forma sí, hermiticidad no |
| **Superposición / combinación lineal** | Σ cᵢ\|i⟩ | Σ attn_i · vᵢ (mezcla de values) | 🟢 misma estructura convexa |
| **Producto tensorial / composición** | H_A ⊗ H_B sistemas compuestos | multi-head (concat espacios) | 🟡 concat ≠ ⊗ pleno |
| **Matriz densidad ρ** | ρ=Σ pᵢ\|ψᵢ⟩⟨ψᵢ\| estado mixto | — (no usado explícitamente) | ⬜ GAP candidato |

**Observación clave:** la atención ES, formalmente, un valor esperado bajo una distribución
de Boltzmann sobre estados (los values), con la energía dada por −QᵀK/√d. Esto NO es metáfora:
es la misma fórmula. Es el ancla literal más fuerte y el punto de partida natural.

---

## 2. GAPS — estructura que MC tiene y los transformers NO explotan

Ordenados por viabilidad estimada (filtro §0), no por atractivo conceptual.

### GAP-A — Matriz densidad / estados mixtos (atención como ρ)
La atención produce una mezcla convexa de values: `out = Σ_i a_i v_i`. En MC eso es
exactamente un **estado mixto** descrito por una matriz densidad ρ. Los transformers tiran
esa estructura: colapsan la mezcla a su media (el vector de salida) y pierden la *coherencia*
(los términos cruzados ⟨v_i\|v_j⟩). 
- **Gap:** la salida de atención retiene solo `Tr(ρ V)` (la media) y descarta toda la
  matriz ρ = Σ a_i \|v_i⟩⟨v_i\|, que codifica la *incertidumbre/dispersión* de la mezcla.
- **Hipótesis útil:** propagar una estadística de segundo orden (varianza de la mezcla, o
  la "pureza" Tr(ρ²)) podría dar a capas superiores señal de *cuán confiada/dispersa* fue
  cada decisión de atención — barato y posiblemente útil para calibración/incertidumbre.
- **Falsable barato:** sí. Calcular Tr(ρ²) por cabeza en GPT-2 y ver si correlaciona con
  algo (entropía de predicción, errores). 

### GAP-B — Evolución unitaria / reversibilidad
En MC la evolución sin medición es **unitaria** (preserva norma e información; es reversible).
Los bloques de transformer NO son unitarios (LayerNorm, ReLU/GeLU, residuales los hacen
contractivos/expansivos e irreversibles).
- **Gap:** redes con capas unitarias/ortogonales tienen gradientes que no explotan ni se
  desvanecen (norma preservada) — relevante para estabilidad en profundidad extrema.
- **Estado del arte:** ya existe trabajo (unitary RNNs, orthogonal init, parametrizaciones
  de Stiefel). NO es virgen. Riesgo de colapsar a literatura existente (como NQP→GPTQ).
- **Falsable:** sí, pero el gap real sobre lo existente es dudoso. Prioridad media-baja.

### GAP-C — Entrelazamiento / no-factorizabilidad como métrica
El **entrelazamiento** mide cuánto un estado compuesto NO factoriza: si \|ψ_AB⟩ ≠ \|ψ_A⟩⊗\|ψ_B⟩.
La entropía de entrelazamiento (von Neumann de la ρ reducida) cuantifica correlación
"genuinamente conjunta".
- **Gap:** no hay una métrica estándar en transformers de cuánto la representación de un
  token está "entrelazada" con otra (más allá del attention weight crudo). La entropía de la
  matriz densidad reducida daría una medida basada en principios.
- **Hipótesis útil:** podría ser una herramienta de *interpretabilidad* (qué tokens forman
  unidades semánticas inseparables) más que de rendimiento.
- **Falsable barato:** sí, medible en activaciones de GPT-2.

### GAP-D — Formalismo de fase / amplitudes complejas
MC vive en ℂ: las amplitudes tienen **fase**, y la interferencia (constructiva/destructiva)
es el mecanismo central. Los transformers son puramente reales — no hay interferencia.
- **Gap:** la atención solo *suma* (interferencia constructiva siempre). No puede *cancelar*
  contribuciones vía fase opuesta. Un mecanismo de atención con valores complejos podría
  representar "este token contradice a aquel" de forma nativa.
- **Estado del arte:** existen complex/phase-aware transformers, resultados mixtos.
- **Falsable:** medio (requiere modificar arquitectura y reentrenar — caro en este setup).

### GAP-E — Principio variacional / energía libre
MC y mecánica estadística minimizan **energía libre** F = E − TS (balance energía/entropía).
La inferencia en transformers no tiene un funcional de energía explícito que se minimice en
el forward pass.
- **Gap/conexión:** los Hopfield networks modernos (Ramsauer et al.) ya mostraron que la
  atención ES la actualización de un modelo de energía tipo Hopfield. Esto vincula atención
  con un paisaje de energía — y MC/estadística tienen herramientas ricas ahí (temperatura,
  transiciones de fase, recocido).
- **Hipótesis útil:** ¿control de "temperatura" dinámico en atención (annealing) mejora
  razonamiento multi-paso? Conecta con el √d como temperatura inversa (§1).
- **Falsable barato:** SÍ — solo modifica el escalado de atención en inferencia, sin
  reentrenar. Alto valor/costo.

---

### GAP-F — Energía libre de Helmholtz (J.P. Chancay, 2026-06-25)
Con $A_i = e^{\beta s_i}/Z$, $s_i = q\cdot k_i/\sqrt{d}$, $\beta = $ temperatura inversa, la
cantidad $F = -\tfrac{1}{\beta}\log Z$ **es** la energía libre de Helmholtz. La atención
computa $Z$ pero nunca usa $F$ ni sus derivadas.
- **Gap:** en física estadística las transiciones de fase se detectan en las *derivadas* de
  $F$ (capacidad calorífica, susceptibilidad). Los transformers ignoran toda esa estructura.
- **Hipótesis:** distintos tipos de cabeza (sintácticas / recuperación de hechos / razonamiento)
  podrían operar en regímenes termodinámicos distintos, algunas cerca de un punto crítico.
- **Falsable barato:** sí — variar $\beta$ en inferencia, medir PPL/entropía/estabilidad,
  buscar picos en $C = \partial\langle E\rangle/\partial T$ y $\chi = \partial\langle A\rangle/\partial T$.

### GAP-G — Información mutua entre cabezas (J.P. Chancay)
Se mide attention entropy y rollout, pero casi nadie mide $I(H_i; H_j)$ entre cabezas. En
mecánica estadística, **correlaciones de largo alcance = indicador de criticidad**.
- **Hipótesis:** cabezas importantes podrían exhibir alta correlación / sincronización en
  ciertos tokens. Permitiría pruning y compresión SIN reentrenar.
- **Falsable barato:** sí (solo medir activaciones).

### GAP-H — Entropía de von Neumann (extensión de GAP-A, J.P. Chancay)
La pureza $\mathrm{Tr}(\rho^2)$ es solo un momento de 2º orden. La entropía de von Neumann
$S(\rho) = -\mathrm{Tr}(\rho\log\rho)$ contiene mucha más información.
- **Hipótesis:** al alucinar / estar inseguro / enfrentar OOD, $S(\rho)$ sube y la pureza
  baja → **estimador de incertidumbre prácticamente gratis**.
- **Falsable barato:** sí.

### GAP-I — Grupo de Renormalización (RG) (J.P. Chancay) — alto potencial, más teórico
Un transformer es una sucesión de mezclas/proyecciones/compresiones $x_l \to x_{l+1}$ que
elimina grados de libertad irrelevantes — exactamente el **coarse-graining del RG de Wilson**.
- **Preguntas:** ¿las capas implementan coarse-graining? ¿hay puntos fijos? ¿clases de
  universalidad entre modelos? Podría explicar las leyes de escala y la profundidad óptima.
- **Falsable:** medio-caro (requiere análisis multi-capa/multi-modelo), Fase 3.

### GAP-J — Geometría de información (J.P. Chancay)
La distribución de atención es una distribución estadística → tiene métrica de Fisher,
curvatura, geodésicas.
- **Hipótesis:** los errores se concentran en regiones de alta curvatura → atención adaptativa
  / temperaturas locales / routing, sin tocar el entrenamiento. (Ojo: NQP ya tropezó con
  Fisher; aquí es Fisher de la *distribución de atención*, objeto distinto y mejor definido.)

### GAP-K — Capacidad calorífica como detector de régimen cognitivo (J.P. Chancay)
$C = \beta^2(\langle E^2\rangle - \langle E\rangle^2)$. En física, picos de $C$ ⇒ transición
de fase. En LLMs podrían marcar cambios de régimen: recuperación de memoria vs razonamiento
multi-hop vs generación creativa. **Muy poco explorado.**
- **Falsable barato:** sí — $C$ es la varianza de la energía, computable de los logits de
  atención que ya existen.

---

## 3. Ranking de viabilidad (candidatos a explorar)

Insight unificador (J.P. Chancay): **toda la termodinámica deriva de la misma $Z$ que la
atención ya computa.** $F$, $\langle E\rangle$, $C$, $\chi$, $S(\rho)$ son funciones de los
mismos logits de atención — medirlas es casi gratis y no requiere reentrenar. Esto eleva todo
el bloque termodinámico (E/F/H/K) a prioridad ALTA conjunta.

| Gap | Coincidencia literal | Gap real vs SOTA | Falsable barato | Fase |
|---|---|---|---|---|
| **GAP-E** (temperatura) | 🟢 softmax=Boltzmann | 🟡 Hopfield conecta | 🟢 sin reentrenar | **1** |
| **GAP-A** (pureza ρ) | 🟢 mezcla=ρ | 🟢 no se usa | 🟢 solo medir | **1** |
| **GAP-H** (von Neumann S) | 🟢 mezcla=ρ | 🟢 no se usa | 🟢 solo medir | **1** |
| **GAP-K** (capacidad calorífica) | 🟢 C=var(E) | 🟢 inexplorado | 🟢 solo medir | **1** |
| **GAP-F** (energía libre) | 🟢 F=−logZ/β | 🟢 derivadas sin uso | 🟢 solo medir | 2 |
| **GAP-G** (info mutua cabezas) | 🟡 medio | 🟢 poco medido | 🟢 solo medir | 2 |
| **GAP-I** (renormalización) | 🟡 coarse-grain | 🟢 explica scaling | 🔴 multi-capa/modelo | 3 |
| **GAP-J** (geom. información) | 🟡 medio | 🟡 (NQP tropezó) | 🟡 | 3 |
| GAP-C (entrelazamiento) | 🟡 | 🟢 | 🟢 | 2 |
| GAP-D (fase compleja) | 🟡 | 🟡 existe | 🔴 reentrenar | baja |
| GAP-B (unitariedad) | 🟡 | 🔴 mucho SOTA | 🟡 | baja |

---

## 4. Plan de experimentos

### Fase 1 (sin reentrenar) — caracterización termodinámica
**EXP-Q01 — Thermodynamic Characterization of Transformer Attention.** Por cabeza de
atención en GPT-2, medir desde los logits de atención $s_i = q\cdot k_i/\sqrt d$:
- temperatura efectiva (dispersión del softmax) vs $\sqrt d$  [E]
- energía libre $F = -\tfrac1\beta\log Z$  [F]
- pureza $\mathrm{Tr}(\rho^2)$  [A]
- entropía de von Neumann $S(\rho)$  [H]
- capacidad calorífica $C = \beta^2(\langle E^2\rangle-\langle E\rangle^2)$  [K]
- **correlacionar todas con:** perplejidad, entropía de salida, (luego) alucinación, OOD.

Pregunta central: ¿hay tipos de cabeza con firmas termodinámicas distintas? ¿Alguna opera
cerca de un punto crítico (picos de $C$/$\chi$)?

### Fase 2 (análisis estructural)
- **EXP-Q02:** información mutua $I(H_i;H_j)$ entre cabezas; correlaciones de largo alcance
  como detector de criticidad → pruning/compresión sin reentrenar  [G]
- barrido de $\beta$ en inferencia: buscar picos en $C$/$\chi$ = transiciones de fase internas [F,K]
- entrelazamiento entre representaciones de tokens [C]

### Fase 3 (más teórico)
- **EXP-Q03:** ¿las capas implementan coarse-graining tipo RG? ¿puntos fijos? ¿universalidad?
  Conexión con leyes de escala y profundidad óptima  [I]
- geometría de información de la distribución de atención  [J]

**Disciplina §0 en cada fase:** medir antes de modificar, refutar barato antes de escalar,
y siempre controlar por el confound obvio (lección de NQP-U1b: correlación parcial, no bivariada).

### Hallazgo incidental (EXP-Q01 debug, 2026-06-25): gradiente de temperatura por profundidad
Los logits de atención de GPT-2 crecen brutalmente con la profundidad (L0 max≈38, L6≈4.9e5,
L11≈6.2e5 — fenómeno conocido de *attention logit growth* / massive activations). En lenguaje
termodinámico: **las capas tienen un gradiente de temperatura efectiva.** Tempranas = calientes
(atención distribuida, mezcla); profundas = frías (estado puro, purity≈1, T_eff≈1, atención
casi determinista). Emerge sin que nadie lo entrene así. Refina GAP-F/K: el cambio de régimen
no es solo entre *tipos de cabeza* sino *a lo largo de la profundidad* — capas finales en estado
puro ≈ "decisión determinista", tempranas ≈ "exploración". Implicación práctica: cualquier
métrica termodinámica debe normalizarse por capa, o la profundidad la domina (confound a controlar).

### Resultado EXP-Q01 (2026-06-25)
- **Estructura de fases confirmada empíricamente.** Régimen termodinámico varía fuerte:
  - Capas tempranas (L0): cabezas "líquidas" (T_eff≈15, purity≈0.43, S_vn≈1.4) Y "congeladas"
    (T_eff≈1.2, purity≈0.99) coexisten en la misma capa.
  - Capas profundas (L11): TODAS cristalizadas (T_eff=1.0, purity=1.0, S_vn=0) — atención
    determinista, colapsada a 1 token.
- **S_vn rastrea incertidumbre de predicción:** corr(S_vn, pred entropy) = −0.20, y
  **sobrevive al control por posición** (parcial −0.18). No es confound (contraste con NQP-U1b).
  → estimador de incertidumbre barato, candidato real (GAP-H).

### Hipótesis de cristalización (J.P. Chancay, 2026-06-25) — NUEVA PRIORIDAD
Si una cabeza tiene T_eff→1 (S_vn→0), su softmax colapsa a δ y la atención deja de ser
promedio ponderado: `Attn ≈ V_{argmax}`. En ese régimen exp/sum/división son **innecesarios**
— se reemplaza por `argmax` (Top-1) o Top-2. Beneficio: menos cómputo, menos memoria (no
guardas la matriz de atención), menos tráfico de memoria (el cuello de botella real de LLMs).

**Forma fuerte — cristalización total:** las capas profundas podrían no necesitar atención;
aproximables por h_{l+1}=f(h_l) con f = MLP / routing / piecewise lineal. La atención profunda
sería solo el mecanismo que *implementa* una decisión discreta ya cristalizada.

**Trampa a vigilar:** S_vn≈0 NO implica reemplazable — ∂Attn/∂q puede seguir importando si el
modelo opera *cerca de una transición* (región sensible). Por eso hay que medir y ablacionar,
no asumir. Métrica de determinismo genuino: R = p₂/p₁ (ratio del 2º al 1er peso). R≪1 ⇒
genuinamente determinista.

### Paper objetivo: *"Thermodynamic Phase Transitions and Attention Crystallization in
### Transformer Networks"*
Ya no es analogía: es un mecanismo concreto de compresión/aceleración derivado de la
termodinámica observada. Resultado central = curva ΔPPL(L) del reemplazo progresivo por Top-k.

### Plan EXP-Q02 — Attention Crystallization

**Pre-registro (J.P. Chancay, 2026-06-25) — fijar hipótesis ANTES de ver datos (anti-NQP):**
- **H0 (nula):** la baja entropía de capas profundas NO permite reemplazo determinista sin
  degradación significativa.
- **H1:** existe un subconjunto de capas profundas cuya atención se aproxima por un operador
  Top-1/Top-k con pérdida de desempeño despreciable y mejora de eficiencia.

**Controles obligatorios (la baja S_vn por sí sola NO basta):**
- **C1 — margen/ratio:** R = p₂/p₁ y margen p₁−p₂. S_vn bajo puede venir de [0.51,0.49,…]
  (no determinista). Cristalización genuina requiere R≪1.
- **C2 — estabilidad al ruido:** P(argmax(q) = argmax(q+ε)). Si el argmax cambia con
  perturbación pequeña, la cabeza está *cerca de una transición* → reemplazarla rompe el
  modelo (la "trampa": determinista ≠ reemplazable).

**Fases:**
- **A (diagnóstico, implementado):** por cabeza/capa medir R y estabilidad-argmax (C1+C2).
- **B (causal):** reemplazo progresivo L11→Top-k, L10-11, … Curva ΔPPL(L) revela el punto de
  congelamiento L_c. Resultado ideal: ΔPPL≈0 hasta cierta profundidad, luego explota.
- **C (forma fuerte):** ¿capas cristalizadas reemplazables por f(h) sin Q/K/atención? →
  transformers híbridos: primeras capas termodinámicas + últimas deterministas.
- **Métricas:** PPL, exact match, throughput, VRAM, latencia.

Si B sale positivo aunque sea en las 2 últimas capas: deja de ser caracterización y pasa a ser
**un mecanismo de aceleración/compresión derivado de una propiedad termodinámica medida** —
resultado publicable mucho más fuerte.

### Resultado EXP-Q02 (2026-06-25) — H1 REFUTADA, pero con hallazgos

**Phase A (diagnóstico):** L1-L11 pasan C1+C2 (R≈0.00-0.05, argmax estable >94%). L0 es la
única "líquida" (R=0.40). El diagnóstico decía: 11 capas cristalizables.

**Phase B (causal):** reemplazo por Top-k REFUTA el diagnóstico:

| reemplazo | Top-1 ΔPPL | Top-2 ΔPPL |
|---|---|---|
| solo L11 | +7.68 | +2.54 |
| L9-11 (3 capas) | +16.5 | +4.71 |
| L3-11 (9 capas) | +65.4 | +12.5 |
| L1-11 | +523 | +78 |
| L0-11 (todas) | +2308 | +500 |

**Conclusión: H0 se sostiene, H1 falla.** Reemplazar incluso SOLO L11 (R=0.00, argmax 100%
estable) por argmax sube PPL +7.7 — no despreciable.

**La lección — C2 (estabilidad argmax) es necesario pero NO suficiente:**
> Aunque p₂/p₁≈0, la atención suma sobre cientos de tokens con pesos minúsculos. Esa **cola
> agrega una contribución sistemática** al value de salida. El argmax captura *dónde* mira la
> cabeza pero descarta la *integración sobre el contexto*. S_vn≈0 mide concentración de *dónde*,
> no irrelevancia del *resto*. Top-2 reduce el daño a 1/3 de Top-1 → la info vive en la cola,
> no en el pico. **Determinista en el peso máximo ≠ reemplazable por selección.**

**Hallazgos positivos (a pesar de refutar H1):**
1. La curva ΔPPL(L) es **suave y monótona, sin codo abrupto** → NO hay punto de congelamiento
   de primer orden; las capas forman un continuo funcional, no fases discretas separables.
   (Refuta la imagen ingenua de "fases discretas líquido/crítico/congelado".)
2. La única transición brusca está en **L0→L1** (Top-1: +523→+2308): L0 es cualitativamente
   distinta (única líquida). La frontera líquido/sólido existe pero está al PRINCIPIO, no al final.
3. Top-2 ≫ Top-1 sistemáticamente → mecanismo de "integración de cola", no de "selección".

**Implicación para el paper:** el ángulo "crystallization → aceleración" no funciona vía
hard top-k. Pero la caracterización termodinámica (Q01) + la refutación cuantitativa de la
selección dura (Q02) + el continuo funcional sí cuentan una historia honesta sobre por qué
la atención NO es comprimible por selección pese a parecer determinista.

### GAP-L — Atención cristalizada con excitaciones residuales (J.P. Chancay, 2026-06-25)
Corrige el ángulo de Q02. La descomposición es **exacta**:
$$\text{Attn} = \underbrace{V_{i^*}}_{\text{cristalizada}} + \underbrace{\epsilon}_{\text{excitaciones}},
\qquad \epsilon = \sum_{i\neq i^*} p_i V_i$$
Q02 mostró que $\epsilon$ NO es despreciable. La pregunta nueva: **¿es $\epsilon$ comprimible?**

**Interpretación física correcta (no "parte imaginaria"):** esto es *estado fundamental +
excitaciones*, $|\psi\rangle = |0\rangle + \epsilon|1\rangle+\dots$, NO $a+ib$. La parte
imaginaria en MC tiene la misma jerarquía que la real y produce interferencia — no es un
término correctivo. El residual SÍ es subdominante. Así que la analogía rigurosa es
ground-state + excitaciones; lo complejo queda como inspiración de representación, no como
afirmación física. (Disciplina anti-NQP: no forzar la analogía.)

**EXP-Q03 — tres modelos de reemplazo en capas profundas:**
- **A:** Top-1 puro (`V_{i*}`) — ya medido en Q02 (rompe).
- **B:** Top-1 + residual completo `V_{i*} + ε` — debe recuperar exacto (sanity).
- **C:** Top-1 + residual low-rank `V_{i*} + λ·r`, r de dimensión d_r ≪ d_head — el test real.
- **Métricas:** ΔPPL vs d_r. Si d_r pequeño recupera casi toda la pérdida → las capas profundas
  son *casi cristalinas con pocas excitaciones* que portan la información. Resultado fuerte.
- **Bonus físico:** |z|²=‖V_{i*}‖²+‖ε‖² y θ=atan(‖ε‖/‖V_{i*}‖) como indicador de proximidad a
  transición / incertidumbre residual.

### Resultado EXP-Q03 (2026-06-25) — Hε REFUTADA (residual NO es low-rank)
Sanity OK (B = baseline exacto, ΔPPL=0). Pero el residual ε no es comprimible:

| dim efectiva ε (90% var) | r=1 | r=4 | r=16 | r=32 |
|---|---|---|---|---|
| **21 / 64** | 33% rec | 46% | 71% | 87% |

El residual necesita ~21/64 dims para 90% de varianza; ni r=32 (medio espacio) recupera >87%
de la pérdida de Top-1. La curva ΔPPL(r) es suave, sin saturación temprana. **La analogía de
teoría efectiva (pocos modos de excitación) NO aplica:** las excitaciones son de alto rango,
más cercanas a *ruido térmico equipartido* que a *fonones de pocos modos*.

### Conclusión unificada de la línea de cristalización (Q02 + Q03)
**La atención de GPT-2 resiste la compresión por toda vía estructural probada:**
- Q02: no por selección dura (Top-k) → la cola de pesos pequeños hace trabajo sistemático.
- Q03: no por proyección low-rank del residual → la cola es de alto rango (~21/64).

La información de la "cola" de atención está **distribuida de forma irreducible** — densa, no
pico + corrección. Resultado negativo fuerte: contra la intuición de que cabezas "deterministas"
(R≈0, S_vn≈0) son comprimibles, la integración sobre el contexto es esencial e incompresible
linealmente. (Posible escape: compresión NO lineal del residual, pero ya fuera del alcance CPU.)

### Priorización y paper objetivo (J.P. Chancay, 2026-06-25)
Balance: 1 positivo robusto (estructura termodinámica) + señal moderada (S_vn↔incertidumbre)
+ 2 negativos fuertes (Q02 selección, Q03 low-rank). En arquitecturas de IA, **negativos bien
establecidos son muy valiosos: eliminan líneas enteras**. Prioridades:
1. **S_vn como estimador de incertidumbre** (bajo riesgo/costo, señal ya existe).
2. **Ley de escala de L_c** (medio, aprovecha lo hecho — EXP-Q04 listo).
3. **Geometría de la cola residual (Dir-D):** PCA mide rango LINEAL; falta **dimensión
   intrínseca** (Isomap/MLE/TwoNN). Si PCA→21 pero intrínseca→5, la conclusión de Q03 cambia
   de "irreducible" a "no lineal pero compresible". **Control necesario para no sobre-afirmar.**
- En pausa: compresión Top-k, reemplazos deterministas, analogías cuánticas fuertes.

**Paper objetivo:** *"Thermodynamic Structure of Transformer Attention: Phase Separation,
Uncertainty Estimation, and the Irreducibility of Residual Context Integration"*. Narrativa:
(1) estructura termodinámica real; (2) S_vn rastrea incertidumbre; (3) capas "cristalizadas"
NO comprimibles por aproximaciones lineales simples; (4) la integración contextual residual
parece propiedad fundamental de la atención.

### EXP-Q05 (Dir-D) — dimensión intrínseca del residual → REENMARCA Q03
Estimar dim intrínseca de ε (TwoNN/MLE, no lineal) vs los 21 lineales de PCA. Califica si la
"irreducibilidad" de Q03 es lineal-only o fundamental. Barato (solo sobre ε ya recolectables).

**Resultado (2026-06-25) — la cola SÍ tiene estructura de baja dimensión (no lineal):**

| | PCA lineal (90% var) | dim intrínseca (TwoNN) |
|---|---|---|
| residual ε (L9-11) | **23 / 64** | **6.8 / 64** |

(TwoNN validado vs swiss-roll: detecta 2.7 donde PCA ve 3 → distingue no-lineal de rango lineal.)

**Corrige la conclusión de Q03:** la cola NO es "irreducible" — es **irreducible LINEALMENTE
pero vive en un manifold no-lineal de ~7D**. La afirmación previa era sobre-afirmación
(lineal-only). Claim correcto: *la integración de contexto es no-lineal pero de baja dimensión*.
**Reabre la compresión** que Q02/Q03 parecían cerrar: el mecanismo correcto es un encoder NO
lineal de la cola (p.ej. MLP 64→~8→64), no truncamiento/proyección lineal.

**Lección (disciplina anti-NQP):** PCA mide rango lineal, no dimensión. "No comprimible por PCA"
≠ "no comprimible". Dir-D (J.P. Chancay) atrapó la sobre-afirmación antes del paper. El paper
debe decir "irreducible linealmente; estructura no-lineal de baja dim" — claim más preciso y
de hecho más interesante (sugiere mecanismo, no solo límite).

### Orden de evidencia (J.P. Chancay, 2026-06-25): Q04-lite ANTES que Q06
Razón metodológica, no computacional: Q06 es mecanístico (¿se puede explotar?), Q04 es validez
externa (¿el fenómeno es robusto a escala?). El encadenamiento sólido para paper es:
**existe el fenómeno → escala → tiene estructura geométrica → puede explotarse.**

**Hipótesis central (la apuesta de mayor valor):** $\dim(M_\epsilon) \approx \text{const}$ — la
dimensión intrínseca del residual contextual es ~independiente del tamaño del modelo. Si cierto,
es una **reducción efectiva de grados de libertad** (análogo a teorías efectivas en física),
mucho más profundo y general que un método de compresión. Escenarios:
- A: cristalización universal + dim_int≈const → Q06 prioridad absoluta + resultado profundo.
- B: dim_int crece lento → Q06 sigue con sentido.
- C: fenómeno desaparece en modelos grandes → Q06 pierde interés.

### EXP-Q04-lite — fenómeno + escala (solo inferencia)
Para gpt2 small/medium/(large): medir L_c (vía R por capa) Y **dim intrínseca del residual**
(TwoNN, el observable clave). Todo forward-only. Discrimina A/B/C.

### Resultado EXP-Q04-lite (2026-06-25) — dim(M_ε) ≈ CONST (robusto, Caso A)
Protocolo controlado (J.P. Chancay): N=1500 pts/cabeza idéntico, 8 cabezas, últimas 3 capas,
mismo dataset, en gpt2 / medium / large (124M→355M→774M, 12→24→36 capas).

| modelo | capas | dim_int | dim_lin (PCA) |
|---|---|---|---|
| gpt2 | 12 | 7.2 ± 1.1 | 31.6 |
| gpt2-medium | 24 | 8.1 ± 0.8 | 30.5 |
| gpt2-large | 36 | 7.4 ± 0.7 | 28.0 |

**spread entre-modelos = 0.9 ≈ ruido entre-cabezas intra-modelo = 0.8.** → La variación con la
escala es indistinguible del ruido natural. El modelo crece 6× en params, 3× en capas, y la
**dimensión intrínseca del residual contextual se queda en ~7-8.** Núcleo geométrico efectivo
casi invariante a escala (Caso A de J.P.) — reducción efectiva de grados de libertad, análogo
a teoría efectiva. (Contraste: dim LINEAL ~30 vs intrínseca ~7 en los tres → estructura
no-lineal de baja dim es universal, no peculiaridad de small.)

**Honestidad — L_c NO es universal:** L_c = 2/1/9 ("MIXED"). large tiene fase líquida extendida
(R: 0.86→…→0 gradual) vs small/medium que cristalizan en L1-L2. La profundidad de cristalización
NO da ley limpia; **el observable universal es dim(M_ε), no L_c.** Decirlo así en el paper.

**Este es el resultado más fuerte del proyecto** y sobrevive control de protocolo (a diferencia
del primer run con N no-controlado que dio 6.7 idéntico = artefacto). Mide validez externa:
el fenómeno geométrico es robusto a escala.

### EXP-Q06 (siguiente) — compresión no-lineal de la cola
Autoencoder 64→32→16→~8→16→32→64 sobre ε; medir reconstrucción y luego reemplazar ε→ε̂ en el
Transformer, medir ΔPPL. Caso A (ΔPPL≈0): cola comprimible no-linealmente (resultado fuerte).
Caso B (degrada): dim geométrica ≠ dim funcional (también valioso). Peligro metodológico
(J.P.): TwoNN da dim intrínseca LOCAL; no garantiza AE global suave de esa dim. Por eso Q06
FALSA la hipótesis H-NL, no la asume.

### GAP-M — ¿Es la cristalización universal o contingente? Ley de escala de L_c (J.P. Chancay)
**Pregunta formal:** ¿la profundidad de cristalización L_c es propiedad universal de la
profundidad, o contingente de arquitectura/tamaño? Escenarios:
1. **Escala con profundidad:** L_c crece con N_layers (modelos profundos → más región cristalina).
2. **Punto fijo (RG):** L_c constante; añadir capas replica el régimen cristalino (→ GAP-I).
3. **Recristalización parcial:** líquido→cristal→mezcla→cristal (cabezas de inducción/recuerdo
   distintas requieren regímenes distintos).
4. **Desaparece en grande:** GPT-2 small cristaliza por baja capacidad (decisiones "duras");
   modelos grandes mantienen mezcla → fenómeno de modelos pequeños, no universal.

**Resultado fuerte buscado:** una ley L_c = f(N_params, N_layers, N_data) — p.ej. L_c ∝ log N,
∝ √N, o constante. Cualquiera sería resultado científico por sí solo, y diría si "estado
fundamental + excitaciones" es general o específico de GPT-2 small.

**EXP-Q04 (escala):** repetir el perfil termodinámico (Q01: T_eff, S_vn, R por capa — BARATO,
solo forward) en GPT-2 small/medium/large. Localizar L_c (capa donde R cae bajo umbral) en cada
uno y ver la tendencia. Restricción de cómputo (CPU): Q01 viable hasta large; Q02/Q03 (decenas
de PPL completas) solo en small. Estrategia: L_c vía firma Q01 barata en 3 tamaños → tendencia.
