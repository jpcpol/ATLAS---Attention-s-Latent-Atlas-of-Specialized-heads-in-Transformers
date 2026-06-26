# NQP — Experiments

## Proposed execution order

### EXP-001 — Diagonal Fisher baseline on GPT-2 small
**Objective:** verify that $\hat{P}$ derived from a diagonal Fisher reduces $\varepsilon_Q$
relative to standard quantization on a controlled model (GPT-2 124M).

**Metrics:** PPL on WikiText-103, per-layer L2 quantization error, effective bits used.

**Comparators:** standard INT8, GPTQ 4-bit, QuIP 4-bit.

**Status:** `src/fisher.py` implemented. Ready to run with `python src/fisher.py --bits 8 --n-calib 256`.

---

### EXP-002 — Scaling to Llama-3 8B
**Prerequisite:** EXP-001 shows improvement on GPT-2.

**Objective:** validate NQP-C1 at practical scale.

**Status:** pending.

---

### EXP-003 — Test of NQP-C2 (strong form)
**Objective:** verify whether an NQP-4bit model beats FP32 on specific tasks.

**Working hypothesis:** on code-review tasks (calibration distribution = code), NQP acts as a
regularizer and improves accuracy vs FP32.

**Status:** pending — requires EXP-002.
