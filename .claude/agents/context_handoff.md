---
name: context_handoff
description: >
  Protocolo de traspaso de contexto entre agentes. No invocar directamente.
model: haiku
effort: low
tools: Read, Write
---

# Protocolo de traspaso de contexto entre agentes

Cuando un agente delega a otro, generá este bloque ANTES de invocar al receptor:

```
## CONTEXTO DEL AGENTE ANTERIOR

**Agente anterior**: @[nombre]
**Tarea completada**: [descripción en una línea]
**Archivos leídos** (no releer):
- [ruta] — [qué información relevante contiene]

**Archivos modificados**:
- [ruta] — [qué cambió y por qué]

**Decisiones tomadas**:
- [decisión] — [razón]

**Lo que necesitás hacer**:
[instrucción clara para el agente receptor]

**Contexto crítico a no perder**:
- [invariante o restricción específica del proyecto]
```

## Reglas

- El bloque reemplaza la necesidad de releer CLAUDE.md desde cero.
- No incluyas info que ya está en MEMORY.md — el receptor lo leerá por su cuenta.
- Si el handoff supera 30 líneas, estás incluyendo demasiado. Resumí.
