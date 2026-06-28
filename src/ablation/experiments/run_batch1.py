"""
Batch-1 driver — vary d_head, measure O_h (with P5 temporal-emergence snapshots).

Orchestrates the approved design (docs/ablation_design.md):
  for each (d_head variant) x (seed):
     train from scratch, and at each snapshot fraction (P5) measure O_h(t) and
     plateau-d_int(t) so the atlas's temporal emergence is recorded;
     after training, run Gate 0; only if it passes does the FINAL O_h count toward the
     §4 predictions (a flat O_h from an immature model is INVALID, not a refutation).

Training (train/) and measurement (measure/) stay separate per the audit; this driver is
the only place they meet. Colab-friendly: pass --device cuda. Writes one JSON per run.
"""

from __future__ import annotations

import os
import sys
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))            # .../src
for p in (_SRC, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train.configs import (ModelCfg, TrainCfg, batch1_variants,
                           scale_control_variants, BATCH1_SEEDS, cfg_to_dict)
from train.train import train_with_snapshots, load_wikitext103_tokens
from measure.atlas_metrics import measure_atlas
from measure.gate0 import gate0


def run_one(model_cfg, train_cfg, seed, *, device, train_ids, val_ids, out_dir,
            save_ckpt=False):
    print(f"\n{'='*72}\n[VARIANT] {model_cfg.tag()} seed={seed}\n{'='*72}")
    emergence = []          # P5: (frac, step, val_loss, O_h, plateau_d_int) per snapshot
    val_curve = []

    def on_snapshot(model, step, frac, val_loss):
        val_curve.append(val_loss)
        # P5 cost control: intermediate snapshots only need the SHAPE of the emergence
        # curve, so they use fewer points (faster); the FINAL snapshot (frac>=1.0) uses
        # full points so the number that feeds P1/P4 and is compared to the cross-arch
        # 0.28/0.20 clusters is measured at the same fidelity as every prior result.
        is_final = frac >= 0.999
        npts = 1200 if is_final else 600
        m = measure_atlas(model, val_ids, device, n_points=npts)
        emergence.append({"frac": round(frac, 3), "step": step, "val_loss": val_loss,
                          "O_h": m["O_h"], "O_h_ci": m["O_h_ci"],
                          "plateau_d_int": m["plateau_d_int"], "n_points": npts})
        oh = m["O_h"]; di = m["plateau_d_int"]
        print(f"      [P5] {frac:>4.0%}  O_h={oh:.3f}  plateau_d_int={di:.2f}  (N={npts})")
        if save_ckpt and out_dir:
            from train.checkpoints import save_snapshot
            save_snapshot(model, os.path.join(out_dir, "ckpt"),
                          model_cfg.tag(), seed, step)

    model, hist = train_with_snapshots(
        model_cfg, train_cfg, seed=seed, device=device,
        train_ids=train_ids, val_ids=val_ids, on_snapshot=on_snapshot)

    passed, g0 = gate0(model, val_ids, device, val_curve=val_curve,
                       vocab_size=model_cfg.vocab_size)
    print(f"\n  [Gate 0] passed={passed}")
    for k in ("G0a_converged", "G0b_depth_regime", "G0c_residual_stable", "G0d_base_atlas"):
        print(f"    {k}: {g0[k]['pass']}")
    # Show the depth profile so 'more steps vs threshold' is decidable from data, not guessed.
    g0b = g0["G0b_depth_regime"]
    prof = "  ".join(f"{p['rel']}:{p['d_int']:.1f}" for p in g0b["profile"])
    print(f"    G0b detail: peak={g0b.get('peak', float('nan')):.2f} "
          f"bump_vs_min={g0b.get('bump_vs_min', float('nan')):.2f} (need >0.5) "
          f"peak_rel={g0b.get('peak_rel', float('nan'))} "
          f"early={g0b.get('peak_early')} | ID profile (rel:d_int): {prof}")

    final = emergence[-1] if emergence else None
    result = {"config": cfg_to_dict(model_cfg, train_cfg), "seed": seed,
              "history": hist, "emergence": emergence, "gate0": g0,
              "final_O_h": (final or {}).get("O_h") if passed else None,
              "final_plateau_d_int": (final or {}).get("plateau_d_int") if passed else None,
              "gate0_passed": passed,
              "VALID": passed}
    if not passed:
        print("  [result] Gate 0 FAILED -> final O_h is INVALID for §4 (not a refutation). "
              "Extend training or exclude.")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f"{model_cfg.tag()}_seed{seed}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote {p}")
    return result


def main():
    ap = argparse.ArgumentParser(description="ATLAS ablation Batch-1")
    ap.add_argument("--mode", choices=["batch1", "scale_control"], default="batch1")
    ap.add_argument("--d-heads", type=int, nargs="+", default=None,
                    help="restrict batch1 to these d_head values (e.g. 64 128); default = all 4")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(BATCH1_SEEDS))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-train-tokens", type=int, default=20_000_000)
    ap.add_argument("--out-dir", type=str, default="../../../docs/ablation_batch1")
    ap.add_argument("--smoke", action="store_true",
                    help="one tiny variant, few steps — validate the pipeline")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip (variant, seed) whose result JSON already exists (default on)")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if the result JSON exists (overrides --resume)")
    ap.add_argument("--save-ckpt", action="store_true",
                    help="save a model checkpoint at each snapshot (for offline re-measure)")
    args = ap.parse_args()

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    print("[data] loading WikiText-103 (incremental) ...")
    train_ids = load_wikitext103_tokens(tok, "train", max_tokens=args.max_train_tokens)
    val_ids = load_wikitext103_tokens(tok, "validation", max_tokens=2_000_000)
    print(f"  train tokens={train_ids.numel():,}  val tokens={val_ids.numel():,}")

    if args.smoke:
        mc = ModelCfg(d_head=64, d_model=256, n_layer=4, seq_len=128)
        tc = TrainCfg(steps=40, batch_size=4, log_every=10,
                      snapshot_fracs=(0.5, 1.0), eval_batches=5,
                      max_train_tokens=args.max_train_tokens)
        run_one(mc, tc, 42, device=args.device, train_ids=train_ids,
                val_ids=val_ids, out_dir="")
        return

    variants = batch1_variants() if args.mode == "batch1" else scale_control_variants()
    if args.d_heads:                          # restrict to requested d_head values
        variants = [mc for mc in variants if mc.d_head in args.d_heads]
        print(f"[filter] restricted to d_head ∈ {args.d_heads} -> "
              f"{[mc.tag() for mc in variants]}")
    tc = TrainCfg(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                  max_train_tokens=args.max_train_tokens)
    summary = []
    for mc in variants:
        for s in args.seeds:
            # Resume-safe: skip a (variant, seed) whose result JSON already exists, so a
            # Colab disconnect never costs more than the one in-flight run. --force re-runs.
            done_path = os.path.join(args.out_dir, f"{mc.tag()}_seed{s}.json")
            if args.resume and not args.force and os.path.exists(done_path):
                with open(done_path, encoding="utf-8") as f:
                    r = json.load(f)
                print(f"\n[SKIP] {mc.tag()} seed={s} — already done "
                      f"(VALID={r.get('VALID')}, O_h={r.get('final_O_h')})")
            else:
                r = run_one(mc, tc, s, device=args.device, train_ids=train_ids,
                            val_ids=val_ids, out_dir=args.out_dir,
                            save_ckpt=args.save_ckpt)
            summary.append({"tag": mc.tag(), "d_head": mc.d_head, "seed": s,
                            "valid": r.get("VALID"), "O_h": r.get("final_O_h"),
                            "plateau_d_int": r.get("final_plateau_d_int")})

    print(f"\n{'='*72}\n[{args.mode.upper()} SUMMARY]\n{'='*72}")
    print(f"  {'d_head':>6} | {'seed':>4} | {'valid':>5} | {'O_h':>6} | {'plateau_d_int':>13}")
    for r in summary:
        oh = f"{r['O_h']:.3f}" if r["O_h"] is not None else "  —  "
        di = f"{r['plateau_d_int']:.2f}" if r["plateau_d_int"] is not None else "  —  "
        print(f"  {r['d_head']:>6} | {r['seed']:>4} | {str(r['valid']):>5} | "
              f"{oh:>6} | {di:>13}")


if __name__ == "__main__":
    main()
