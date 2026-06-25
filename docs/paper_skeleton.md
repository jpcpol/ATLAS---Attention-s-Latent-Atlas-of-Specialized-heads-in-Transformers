# Paper Skeleton — Thermodynamic Structure and Intrinsic Geometry of Residual Attention in Transformers

**Estado:** Esqueleto · 2026-06-25
**Autores:** Juan Pablo Chancay, Claude (Opus 4.8 / Sonnet 4.6)
**Origen:** línea sucesora de NQP. Ver `theory/quantum_transformer_map.md` para el mapa completo
de gaps y `theory/uncertainty_principle.md` para la rama refutada (NQP-U).

---

## One-liner (nivel abstract) — claim mínimo válido (NO fiber-bundle formal)

> We show that residual attention in GPT-2 is structured as a **collection of head-specific
> nonlinear manifolds with consistent intrinsic dimensionality (~7), embedded in mutually
> non-aligned subspaces**, yielding an apparent higher-dimensional global representation (~17D).
> Each head's contextual residual occupies its own low-dimensional manifold (intrinsic dim
> scale-invariant within the GPT-2 family), and these manifolds are **not coordinatizable by a
> shared linear system** (inter-head subspace overlap ≈ 0.28; head-centering does not collapse
> the union, ~12.7D). This coexists with a thermodynamic phase structure that depends on model
> capacity, separating a scale-invariant per-head geometric core from macroscopic,
> training-dependent observables.

**Título:** *"Non-Aligned Manifold Atlas in Transformer Residual Attention"*. El resultado
central es el **desacoplamiento geométrico entre cabezas**: representaciones locales ~7D
consistentes pero sin coordinación global lineal.

**LÍMITE EXPLÍCITO (anti-overclaim, J.P. Chancay):** "atlas" se usa en sentido descriptivo
(familia de cartas locales), NO como **fiber bundle formal**. NO afirmamos: mapas de transición
suaves entre cartas, consistencia diferenciable global, ni estructura base+fibra. Eso requeriría
medir continuidad entre cabezas (no solo overlap) y estructura diferenciable — trabajo futuro.
Lo medido: dimensión local consistente + no-alineación de subespacios + no-colapso por recentrado.

## Abstract (borrador, nivel workshop)

> We show that residual attention in GPT-2 forms a **scale-invariant, low-dimensional
> nonlinear constraint manifold (~7D)** — an *emergent constraint manifold* in deep
> representations — coexisting with a thermodynamic phase structure that depends on model
> capacity. This separates geometric invariants (the residual manifold) from macroscopic,
> training-dependent observables (crystallization depth L_c). We rule out the obvious
> linear explanations (sparsity, low-rank) and find that what remains is a geometric
> constraint. Scale-invariance is established **within the GPT-2 family / a fixed training
> distribution**, not across architectures.

## Tesis central

El residuo contextual de la atención profunda (ε = Attn − V_{i*}) es un **emergent constraint
manifold**: una variedad no-lineal de dimensión efectiva ≈ 7-8 (vs ~30 lineal), cuya dimensión
es **invariante a la escala del modelo dentro de la familia GPT-2**. Coexiste con una estructura
termodinámica de fases que **sí depende del régimen de capacidad** (L_c). Dos niveles separables:

> **Macro (termodinámico, dependiente de escala/capacidad):** régimen de fase, T_eff, L_c.
> **Micro (geométrico efectivo, escala-invariante en la familia):** manifold residual ~7D.

Consecuencia: *el escalamiento del modelo no cambia la geometría efectiva del residuo; solo
desplaza dónde aparece el régimen de baja temperatura.* El resultado central es la **existencia**
del manifold no-lineal (Q05); su **estabilidad a escala** (Q04) es la propiedad que lo eleva.

**Wording de protección (anti-overclaim):** "universal" → **"scale-invariant within the GPT-2
family / a fixed training-distribution family"**. No se prueba cross-architecture.

---

## Estructura de dos capas (la contribución conceptual)

| | Capa A — Termodinámica (macro) | Capa B — Geometría efectiva (micro) |
|---|---|---|
| Objeto | dinámica de atención | espacio de estados residuales |
| Observables | fases, T_eff, S_vn, L_c | dim intrínseca, no-linealidad |
| Escala | dependiente (parcial) | **invariante** |
| Evidencia | Q01, Q02, Q04-lite | Q03, Q05, Q04-lite |

---

## Resultados (lo que YA está justificado, Q01–Q05)

Orden de fuerza conceptual (no de número): **R-MANIFOLD (Q05) es el corazón** — sin la
existencia del objeto no-lineal, su estabilidad no significa nada. Luego su estabilidad (Q04),
el contraste lineal (Q02/Q03) que lo motiva, y la termodinámica (Q01) como capa macro + bridge.

### R-ATLAS (corazón) — Manifolds por-cabeza no-alineados (Q05, Q05d)
Cada cabeza tiene residual de dim intrínseca (TwoNN) ≈ 7, frente a ~30 lineal (PCA) — effective
DOF ≪ embedding dim, un manifold local por cabeza. **Pero NO comparten coordenadas:** overlap de
subespacios entre cabezas = 0.28 (≈ortogonal), y el pooled centrado por-cabeza no colapsa (12.7D).
→ el sistema es una **colección de variedades locales ~7D embebidas en subespacios mutuamente
no-alineados**, unión ≈17D. Cuatro nulas descartadas DEFINITIVAMENTE:
NO single-manifold (overlap≠1), NO global low-rank (Q03), NO shared latent basis (overlap 0.28),
NO pure gauge/offset (dim centrada 12.7≠7).
(TwoNN validado: 2.7 en swiss-roll donde PCA ve 3; overlap validado: 0.0 atlas vs 1.0 shared en
sintéticos.) **El claim central es el desacoplamiento geométrico entre cabezas.** "Atlas" en
sentido descriptivo (cartas locales), NO fiber-bundle formal (sin mapas de transición medidos).

Modelo mínimo consistente: ε = ⋃_h M_h, con dim(M_h)≈7, embeddings no-alineables, unión ~12-17D.

### R-STABILITY — Dimensión Y no-alineación son escala-invariantes (Q04-lite, Q05d-scale)
Protocolo controlado (N idéntico, mismas cabezas/capas) en gpt2/medium/large (124M→774M):

| modelo | capas | dim_int | dim_lin | inter-head overlap | centered_dim |
|---|---|---|---|---|---|
| small | 12 | 7.2 ± 1.1 | 31.6 | 0.285 | 7.1 |
| medium | 24 | 8.1 ± 0.8 | 30.5 | 0.282 | 6.7 |
| large | 36 | 7.4 ± 0.7 | 28.0 | 0.281 | 5.7 |

**Dos invariantes a escala:** (a) dim intrínseca ~7 (σ_entre-modelos 0.9 ≈ σ_entre-cabezas 0.8);
(b) **no-alineación inter-cabeza overlap ≈ 0.28, spread 0.004 a través de 6× params.** Ambos
ELEVAN R-ATLAS: el atlas de manifolds no-alineados no solo existe, es estable a escala (within
GPT-2 family). centered_dim ≈ dim_int por modelo confirma que el ~12.7 de Q05d venía de mezclar
capas; con 1 capa controlada, la no-alineación es entre cabezas de la misma capa.
**Matiz honesto:** dim_int decrece levemente y monótono (6.7→6.1→5.3, dentro del ruido ±1.1);
la no-alineación NO (es plana). El invariante robusto es el overlap.

### R-LINEAR — Por qué el manifold es no-lineal: refutamos lo obvio (Q02, Q03) [negativos que fortalecen]
- Q02: reemplazo por Top-k (selección dura) destruye PPL (+7.7 solo en L11). Descarta la
  **sparsity-compression story**: la atención es integración, no selección.
- Q03: ε necesita ~30 dims LINEALES (PCA 90%); low-rank no comprime (r=32 → 87%). Descarta la
  **low-rank story**.
- Narrativa: *descartamos las explicaciones obvias (sparsity, low-rank); lo que queda es una
  restricción geométrica no-lineal* — patrón de papers sólidos de ML theory.

### R-THERMO (capa macro + bridge) — Estructura de fases y S_vn como proxy de incertidumbre (Q01)
softmax(QKᵀ/√d) = distribución de Boltzmann ⇒ F, ⟨E⟩, C, S_vn de la misma Z. Gradiente de
T_eff por profundidad: tempranas líquidas (T_eff≈15), profundas cristalizadas (T_eff≈1).
**Bridge result (elevar, no enterrar):** la entropía de von Neumann S_vn(ρ) es un **proxy
calibrado de incertidumbre predictiva** a través de cabezas y profundidad (corr −0.20, sobrevive
control por posición, parcial −0.18) — un *metric candidate*, no solo una observación.

### R-Lc — L_c NO es escala-invariante (Q04-lite) [delimitación honesta, reconcilia macro/micro]
L_c = 2/1/9; large tiene fase líquida extendida. L_c = f(capacidad, entrenamiento, arquitectura).
**El invariante es geométrico (dim M_ε), no termodinámico-macro (L_c).** Dos invariantes de
naturaleza distinta — núcleo de la separación macro/micro.

---

## ¿Qué ES el atlas de manifolds ~7D? (competing interpretations)

El atlas refina la pregunta: cada cabeza tiene su propio manifold ~7D con coordenadas propias.
¿Qué son esas ~7 dim POR CABEZA? Hipótesis en competencia (no duda vaga):
1. **Head-specific semantic modes:** cada cabeza codifica ~7 funciones semánticas propias; la
   incompatibilidad de coordenadas = especialización funcional (cada cabeza "mira" cosas
   distintas). Coherente con interpretabilidad: cabezas como detectores especializados.
2. **Head-specific retrieval modes:** ~7 modos de recuperación contextual por cabeza.
3. **Optimization-induced:** el atlas como producto de la dinámica de entrenamiento (cabezas
   desacopladas para minimizar interferencia), no semánticamente interpretable per se.

El que las coordenadas sean incompatibles (overlap 0.28) FAVORECE 1/3 sobre una base compartida:
las cabezas no son rotaciones de un mismo manifold, son módulos geométricamente desacoplados.

## Posicionamiento vs literatura (para related work)

- NO es mechanistic interpretability clásica (no es circuitos/features individuales).
- NO es scaling-laws clásico (no es loss vs compute; es geometría vs escala).
- NO es compression research clásico (refutamos sparsity y low-rank explícitamente).
- ES una tesis híbrida: **geometría efectiva emergente + dinámica de atención**. Conecta con
  "intrinsic dimension of representations" y "emergent constraint manifolds" en rep. learning.
- Defensa anticipada al ataque "¿es solo dimensionalidad aplicada a activaciones?": la respuesta
  es la *combinación* — invarianza a escala (R-STABILITY) + refutación de lo lineal (R-LINEAR)
  + bridge con incertidumbre (R-THERMO). Ninguna métrica aislada; un sistema de evidencia.

## Lo que NO se puede afirmar todavía (honestidad / trabajo futuro)

- **Explotabilidad funcional** del manifold (Q06 — autoencoder). Pendiente.
- **Regularidad geométrica (Q05b/c/d, 2026-06-25):** M_ε es conexo (1 comp), homogéneo,
  localmente plano (interp 0.60), no se fragmenta por capa (ratio 5.0). **Q05d (atlas test)
  resuelve la naturaleza del objeto:** subspace overlap entre cabezas = 0.28 (≈ortogonal) y dim
  centrada-por-cabeza = 12.7 (no colapsa a 7). → **manifolds por-cabeza NO-ALINEADOS confirmados**,
  descartando "solo offsets" (que daría overlap≈1, dim centrada≈7). Coordenadas latentes
  incompatibles. ("Atlas" descriptivo, no fiber-bundle formal — falta medir mapas de transición.)
- Generalización fuera de la familia GPT-2 (Llama, Mistral): no probada.

---

## Claim preciso (no sobre-afirmar)

NO: "universalidad del modelo completo".
SÍ: **"universalidad del subespacio de excitaciones residuales profundas"** — más específico y
más fuerte. El núcleo geométrico ~7D es del residuo, no del modelo entero.

---

## Plan de cierre

- **Fase 1 (completa):** Q01–Q05.
- **Fase 2 (clarificación geométrica, antes de afirmar explotabilidad):** Q05b smoothness /
  interpolación; Q05c clustering (¿submanifolds por capa/tipo de cabeza/posición?).
- **Fase 3 (opcional, alto impacto):** Q06 autoencoder funcional — solo si Q05b/c muestran
  que M_ε es globalmente regular.
