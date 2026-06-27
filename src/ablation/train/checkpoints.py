"""
Checkpoint I/O for the ablation harness.

Snapshots are saved at the pre-registered training fractions (P5) so the temporal
emergence of the atlas can be measured offline without re-training. We save the bare
state_dict plus the config tag and step, keeping files small and reload trivial.
"""

from __future__ import annotations

import os
import torch


def ckpt_path(out_dir: str, tag: str, seed: int, step: int) -> str:
    return os.path.join(out_dir, f"{tag}_seed{seed}_step{step}.pt")


def save_snapshot(model, out_dir: str, tag: str, seed: int, step: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = ckpt_path(out_dir, tag, seed, step)
    torch.save({"state_dict": model.state_dict(), "tag": tag, "seed": seed, "step": step}, p)
    return p


def load_into(model, path: str, device="cpu"):
    blob = torch.load(path, map_location=device)
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model.load_state_dict(sd)
    model.to(device)
    return model
