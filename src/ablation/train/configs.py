"""
Ablation configs — the matched-scale variant grid and training hyperparameters.

Implements docs/ablation_design.md (APPROVED). Batch-1 varies ONLY the (d_head, n_heads)
head partition of a fixed d_model; verified matched-scale: all variants have identical
parameter counts because Q/K/V/O projections are d_model x d_model regardless of the head
split. configs here are plain dataclasses so train/experiments code stays declarative.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class ModelCfg:
    """One ablation variant. d_head is the intervened axis; n_head is derived."""
    d_head: int
    d_model: int = 512
    n_layer: int = 8
    seq_len: int = 256              # also n_positions; measurement must not exceed it
    vocab_size: int = 50257

    @property
    def n_head(self) -> int:
        assert self.d_model % self.d_head == 0, (self.d_model, self.d_head)
        return self.d_model // self.d_head

    def tag(self) -> str:
        return f"dhead{self.d_head}_dmodel{self.d_model}_L{self.n_layer}"


@dataclass(frozen=True)
class TrainCfg:
    steps: int = 6000
    batch_size: int = 16
    lr: float = 3e-4
    warmup: int = 300
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # P5: fractions of training at which to snapshot + measure (atlas temporal emergence).
    snapshot_fracs: tuple = (0.1, 0.25, 0.5, 0.75, 1.0)
    eval_seed: int = 0             # FIXED across variants so val_loss is comparable
    eval_batches: int = 40
    log_every: int = 200
    max_train_tokens: int = 20_000_000

    def snapshot_steps(self) -> list:
        """Absolute step indices for the pre-registered snapshot fractions."""
        steps = sorted({max(1, round(f * self.steps)) for f in self.snapshot_fracs})
        return steps


# ---- the pre-registered Batch-1 grid -------------------------------------------------

def batch1_variants():
    """4 d_head variants at the same d_model — the (d_head, n_heads) package intervention."""
    return [ModelCfg(d_head=dh) for dh in (32, 64, 128, 256)]


def scale_control_variants():
    """P3: fixed d_head=64, two scales (d_model 512 vs 768) — the fixed-point-like check."""
    return [ModelCfg(d_head=64, d_model=512), ModelCfg(d_head=64, d_model=768)]


BATCH1_SEEDS = (42, 123)


def cfg_to_dict(model_cfg: ModelCfg, train_cfg: TrainCfg) -> dict:
    d = {"model": asdict(model_cfg), "train": asdict(train_cfg)}
    d["model"]["n_head"] = model_cfg.n_head
    d["train"]["snapshot_steps"] = train_cfg.snapshot_steps()
    return d
