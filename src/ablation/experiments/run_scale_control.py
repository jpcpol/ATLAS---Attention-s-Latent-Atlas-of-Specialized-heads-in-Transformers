"""
Scale-control driver (P3) — the fixed-point-like check, the most important control.

Holds d_head = 64 FIXED and varies scale (d_model 512 vs 768) at the same depth. If O_h is
invariant to scale at fixed d_head (within the cross-arch robustness band < 0.02), that is
the cleanest evidence the design can produce for fixed-point-like behavior. If O_h jumps,
the régime index is not d_head alone.

This is just run_batch1 in --mode scale_control; kept as a separate entry point so the P3
control is run and reported on its own, with its own pre-registered pass band.
"""

from __future__ import annotations

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_batch1 import main as _main


if __name__ == "__main__":
    # default to scale_control mode unless the user overrides --mode
    if "--mode" not in sys.argv:
        sys.argv += ["--mode", "scale_control"]
    if "--out-dir" not in sys.argv:
        sys.argv += ["--out-dir", "../../../docs/ablation_scale_control"]
    _main()
