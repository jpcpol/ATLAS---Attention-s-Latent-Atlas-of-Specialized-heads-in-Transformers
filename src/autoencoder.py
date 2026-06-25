"""
NQP-Q — Per-head nonlinear autoencoder on the attention residual (EXP-Q06).

The whole geometric chain (Q03→Q05→Q05d) implies a falsifiable functional claim:
if each head's residual ε lives on a ~7D NONLINEAR manifold (TwoNN ≈ 7) embedded in a
~30D linear span (PCA), then a PER-HEAD nonlinear autoencoder 64→d→64 should
reconstruct ε — and recover the PPL that hard Top-1 destroys — far better than the
LINEAR rank-d projection of Q03 at the same bottleneck d.

This is the decisive test of the manifold reading, and it is the test the atlas
result constrains: because heads use incompatible coordinates (O_h ≈ 0.28), the
autoencoder MUST be per-head, not a single global one. We therefore train one tiny
AE per (layer, head) on collected residuals, then patch attention to replace
ε → ε̂ = AE(ε) on the deep layers and measure ΔPPL.

Comparison (the headline of Q06):
    Q03 linear  P_r P_rᵀ ε      vs   Q06 nonlinear  AE_d(ε)      at equal d.
If AE ≫ PCA  → the nonlinear manifold is real AND exploitable.
If AE ≈ PCA  → the TwoNN ~7 is a local artifact; no usable nonlinear structure.
"""

from __future__ import annotations

import math
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from torch import nn, Tensor

from crystallize import perplexity
from residual import collect_residual_svd, ResidualPatch
from intrinsic import collect_residuals


# ---------------------------------------------------------------------------
# the per-head autoencoder
# ---------------------------------------------------------------------------

class HeadAE(nn.Module):
    """Tiny MLP autoencoder dh -> d -> dh with a nonlinear bottleneck."""
    def __init__(self, dh: int, d: int, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dh, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.dec = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, dh))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_head_ae(E: Tensor, d: int, epochs: int = 300, lr: float = 1e-3,
                  hidden: int = 64, seed: int = 0) -> tuple[HeadAE, float]:
    """Train an AE on residuals E [N, dh]. Returns (model, fraction-variance-unexplained)."""
    torch.manual_seed(seed)
    dh = E.shape[1]
    ae = HeadAE(dh, d, hidden)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    # split for honest reconstruction error
    n = E.shape[0]; ntr = int(n * 0.8)
    perm = torch.randperm(n)
    tr, te = E[perm[:ntr]], E[perm[ntr:]]
    var = te.var(0).sum().clamp(min=1e-12)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (ae(tr) - tr).pow(2).sum(-1).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        fvu = ((ae(te) - te).pow(2).sum(-1).mean() / var).item()   # frac var unexplained
    return ae, fvu


# ---------------------------------------------------------------------------
# patched attention: V_{i*} + AE(ε)  (per head)
# ---------------------------------------------------------------------------

def make_ae_attn(orig_module, layer_idx, aes, n_head):
    def forward(hidden_states, **kwargs):
        B, T, D = hidden_states.shape
        dh = D // n_head
        qkv = orig_module.c_attn(hidden_states)
        q, k, v = qkv.split(D, dim=2)
        shp = lambda x: x.view(B, T, n_head, dh).permute(0, 2, 1, 3)
        q, k, v = shp(q), shp(k), shp(v)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
        mask = torch.tril(torch.ones(T, T, device=hidden_states.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        p = scores.softmax(dim=-1)
        ctx_full = p @ v
        i_star = p.argmax(dim=-1)
        v_star = torch.gather(v, 2, i_star.unsqueeze(-1).expand(-1, -1, -1, dh))
        eps = ctx_full - v_star
        outs = []
        for h in range(n_head):
            ae = aes[(layer_idx, h)]
            with torch.no_grad():
                eh = ae(eps[:, h].reshape(-1, dh)).reshape(B, T, dh)
            outs.append(v_star[:, h] + eh)
        ctx = torch.stack(outs, dim=1).permute(0, 2, 1, 3).reshape(B, T, D)
        return (orig_module.c_proj(ctx), None)
    return forward


class AEPatch:
    def __init__(self, model, layers, aes, n_head):
        self.model, self.layers, self.aes, self.n_head = model, layers, aes, n_head
        self._orig = {}

    def __enter__(self):
        for li in self.layers:
            attn = self.model.transformer.h[li].attn
            self._orig[li] = attn.forward
            attn.forward = make_ae_attn(attn, li, self.aes, self.n_head)
        return self

    def __exit__(self, *a):
        for li, fwd in self._orig.items():
            self.model.transformer.h[li].attn.forward = fwd
        self._orig.clear()


# ---------------------------------------------------------------------------
# EXP-Q06
# ---------------------------------------------------------------------------

def run_exp_q06(deep_layers=(9, 10, 11), d_bottleneck=7, d_compare=(7,),
                n_blocks=30, epochs=300, device="cpu", seed=42):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset
    import os, statistics
    torch.manual_seed(seed)
    if str(device) == "cpu":
        torch.set_num_threads(min(8, os.cpu_count() or 4))

    print("[EXP-Q06] Loading GPT-2 small...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2"); model.eval()
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in ds["validation"]["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"].squeeze(0)
    seq_len = 256
    n_head = model.config.n_head
    dh = model.config.n_embd // n_head

    ppl_fp = perplexity(model, ids, seq_len, device, n_blocks)
    print(f"  PPL baseline (full softmax) = {ppl_fp:.3f}")

    # collect residuals (for AE training) and SVD basis (for the linear Q03 baseline)
    print(f"[EXP-Q06] Collecting residuals + SVD basis for layers {deep_layers}...")
    res = collect_residuals(model, ids, seq_len, device, deep_layers, n_blocks=8, max_points=4000)
    basis, _ = collect_residual_svd(model, ids, seq_len, device, deep_layers, n_blocks=8)

    # Top-1 (the damage we are trying to repair) and full (sanity)
    with ResidualPatch(model, deep_layers, basis, 0, "top1"):
        ppl_top1 = perplexity(model, ids, seq_len, device, n_blocks)
    print(f"  [Top-1 pure] PPL = {ppl_top1:.3f}  ΔPPL={ppl_top1-ppl_fp:+.3f} (damage to repair)")

    def recovered(ppl):  # % of Top-1 loss recovered
        return (ppl_top1 - ppl) / max(ppl_top1 - ppl_fp, 1e-9) * 100

    # train one AE per head at the bottleneck dimension
    print(f"\n[EXP-Q06] Training per-head AEs (dh={dh} -> {d_bottleneck} -> dh), "
          f"{epochs} epochs each...")
    aes, fvus = {}, []
    for key in sorted(res):
        ae, fvu = train_head_ae(res[key], d_bottleneck, epochs=epochs, seed=seed)
        aes[key] = ae; fvus.append(fvu)
    print(f"  mean reconstruction FVU (held-out): {statistics.mean(fvus):.3f} "
          f"(0=perfect, 1=useless)")

    with AEPatch(model, deep_layers, aes, n_head):
        ppl_ae = perplexity(model, ids, seq_len, device, n_blocks)
    print(f"  [Q06 nonlinear AE  d={d_bottleneck}] PPL = {ppl_ae:.3f}  "
          f"ΔPPL={ppl_ae-ppl_fp:+.3f}  recovered={recovered(ppl_ae):.1f}%")

    # head-to-head vs LINEAR PCA (Q03) at the same bottleneck(s)
    print(f"\n[EXP-Q06] Linear PCA baseline (Q03) at equal d — the decisive comparison:")
    print(f"  {'d':>3} | {'PCA recovered':>13} | {'AE recovered':>12}")
    comparison = {}
    for d in d_compare:
        with ResidualPatch(model, deep_layers, basis, d, "lowrank"):
            ppl_pca = perplexity(model, ids, seq_len, device, n_blocks)
        rec_pca = recovered(ppl_pca)
        # AE recovered only available at d_bottleneck; mark others as n/a
        rec_ae = recovered(ppl_ae) if d == d_bottleneck else float("nan")
        comparison[d] = (rec_pca, rec_ae)
        ae_str = f"{rec_ae:11.1f}%" if not math.isnan(rec_ae) else "        n/a"
        print(f"  {d:>3} | {rec_pca:12.1f}% | {ae_str}")

    print(f"\n{'='*66}\n[EXP-Q06 VERDICT] is the per-head nonlinear manifold EXPLOITABLE?\n{'='*66}")
    rec_pca_at_d = comparison.get(d_bottleneck, (float('nan'), None))[0]
    rec_ae_at_d = recovered(ppl_ae)
    print(f"  at d={d_bottleneck}:  PCA recovers {rec_pca_at_d:.1f}%   AE recovers {rec_ae_at_d:.1f}%")
    if rec_ae_at_d > rec_pca_at_d + 10 and rec_ae_at_d > 50:
        print(f"  => NONLINEAR STRUCTURE IS REAL & EXPLOITABLE: a per-head AE at d={d_bottleneck} "
              f"beats linear PCA by {rec_ae_at_d-rec_pca_at_d:.0f}pp. The ~7D manifold can be used; "
              f"the atlas reading (per-head, not global) is functionally validated.")
    elif rec_ae_at_d > rec_pca_at_d + 5:
        print(f"  => MODEST NONLINEAR GAIN: AE beats PCA by {rec_ae_at_d-rec_pca_at_d:.0f}pp. "
              f"Some exploitable nonlinear structure, but not dramatic.")
    else:
        print(f"  => NO USABLE NONLINEAR GAIN: AE ≈ PCA at d={d_bottleneck}. The TwoNN ~7 reflects "
              f"LOCAL curvature without a globally exploitable low-dim parametrization. Honest "
              f"negative: the manifold is real geometrically but not (yet) functionally compressible.")
    return {"ppl_fp": ppl_fp, "ppl_top1": ppl_top1, "ppl_ae": ppl_ae,
            "rec_ae": rec_ae_at_d, "comparison": comparison, "mean_fvu": statistics.mean(fvus)}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="NQP-Q EXP-Q06 — per-head nonlinear AE on residual")
    p.add_argument("--layers", type=int, nargs="+", default=[9, 10, 11])
    p.add_argument("--d", type=int, default=7, help="AE bottleneck dimension")
    p.add_argument("--d-compare", type=int, nargs="+", default=[7])
    p.add_argument("--n-blocks", type=int, default=30)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()
    run_exp_q06(deep_layers=tuple(args.layers), d_bottleneck=args.d,
                d_compare=tuple(args.d_compare), n_blocks=args.n_blocks,
                epochs=args.epochs, device=args.device)
