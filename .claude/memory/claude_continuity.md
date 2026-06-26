---
name: claude-continuity
description: Protocolo de arranque para Claude en NQP / ATLAS
metadata:
  type: reference
---

# Protocolo de arranque — NQP / ATLAS

## Al iniciar una sesión

1. Leer `.claude/MEMORY.md` — índice de 4 líneas
2. Leer `.claude/memory/session_state.md` — pendientes de la última sesión
3. `CLAUDE.md` raíz ya se carga automáticamente
4. Para contexto científico profundo: `docs/paper_draft.md` §0 y §3

## Dónde está todo

| Qué | Dónde |
| --- | --- |
| Contexto del proyecto, estructura, convenciones | `CLAUDE.md` raíz |
| Preprint completo (frozen) | `docs/paper_draft.md` |
| Preguntas de investigación abiertas | `docs/research_questions.md` |
| Cierre del arco NQP-C1 | `docs/retrospective_vs_original_goal.md` |
| Formalización matemática | `theory/` |
| Experimentos cuantización (cerrados) | `experiments/` |
| Implementación activa | `src/` |
| Estado operativo del proyecto | `.claude/memory/project_state.md` |
| Decisiones de investigación (pivots, metodología) | `.claude/memory/research_decisions.md` |

## Contexto crítico a no perder

- El proyecto se llama NQP en disco pero científicamente es ATLAS
- NQP-C1 (cuantización) está **refutada** — no reabrir sin evidencia nueva
- NQP-U1a ✅ (bases no conmutan, ángulo ≈49°) pero U1b ❌ (sin consecuencia operativa)
- Resultado central: O_h ≈ 0.28, scale-invariant, corpus-invariant — **frozen, paper completo**
- La pregunta abierta central: **¿la geometría es causal o meramente descriptiva?**
- Modelos: GPT-2 family vía `transformers`. Datos: WikiText-103 + C4

## Política de actualización

- `session_state.md` — al final de cada sesión con archivos tocados y pendientes
- `project_state.md` — cuando un experimento cambia de estado
- `research_decisions.md` — cuando se toma un pivot o decisión metodológica importante
- `docs/paper_draft.md` — NUNCA modificar sin instrucción explícita del usuario (está frozen)
- Wiki Obsidian — solo en resultados de envergadura (nuevo finding, paper enviado)
