---
name: nqp-project-state
description: Estado actual del proyecto NQP/ATLAS — experimentos y backlog
metadata:
  type: project
---

# Estado del proyecto — NQP / ATLAS

**Última actualización**: 2026-06-26

## Rama cuantización (CERRADA)

| Experimento | Estado | Resultado |
| --- | --- | --- |
| EXP-001: diagonal Fisher (src/fisher.py) | ❌ Refutado | P̂ = I (sin rotación útil) |
| A-G4: block-wise Fisher (src/fisher_block.py) | ❌ Refutado | No supera GPTQ+AWQ+QuIP |
| NQP-C1 (hipótesis central) | ❌ Refutada | Fisher de activaciones rank ~2 |
| NQP-C2 (forma fuerte) | ❌ Congelada | Dependía de C1 |

## Rama incertidumbre (PARCIALMENTE CERRADA)

| Experimento | Estado | Resultado |
| --- | --- | --- |
| EXP-U01: commutator [P̂_W, P̂_A] (src/uncertainty.py) | 🟡 En ejecución | U1a ✅ ángulo ≈49° (vs 83° random) |
| EXP-U02: Pareto ε_W/ε_A (src/pareto.py) | ⬜ Pendiente U1a | U1b ❌ correlación espuria (−0.04) |
| NQP-U1 (principio incertidumbre) | ⚠️ Parcial | U1a real, sin consecuencia operativa |

## Rama geometría / ATLAS (RESULTADO CENTRAL)

| Componente | Estado |
| --- | --- |
| Atlas scale-invariant (O_h ≈ 0.28) | ✅ Frozen — resultado central |
| Manifold ~7D por head (TwoNN, src/intrinsic.py) | ✅ Robusto |
| Termodynamics (Boltzmann, src/thermo.py) | ✅ Bridge result |
| Compresión funcional EXP-Q06 (src/autoencoder.py) | ❌ AE ≈ PCA — negativo documentado |
| Preprint (docs/paper_draft.md) | ✅ Completo, frozen |

## Pendientes activos (backlog)

| ID | Tarea | Prioridad |
| --- | --- | --- |
| P-001 | Confirmación bibliográfica (referencias [1]–[16]) | Alta |
| P-002 | Cross-architecture: Llama / Mistral (promovería "GPT-2" → "scale-invariant general") | Media |
| P-003 | Geometric routing across heads (MoE por geometría, no logits) | Baja — especulativo |
| P-004 | Métricas diagnósticas con O_h (detección de head collapse) | Baja |
| P-005 | Atlas stability under fine-tuning (pretrain → instruct → RLHF) | Baja |

## Pregunta central abierta

**¿La geometría de ATLAS es causal o meramente descriptiva?**
Determina el valor aplicado del proyecto. Sin respuesta hasta P-002+.
