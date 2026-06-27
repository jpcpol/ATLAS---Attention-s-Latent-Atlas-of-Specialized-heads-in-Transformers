"""
Training loop for the ablation harness — training ONLY (no measurement here).

Per docs/ablation_design.md. Trains one variant from scratch on WikiText-103, yielding
the model at each pre-registered snapshot step (P5) so the experiment driver can measure
the atlas's temporal emergence without re-training.

Fixes from the audit of the monolith:
  S1 — eval uses a FIXED eval seed (TrainCfg.eval_seed) for every variant, so val_loss is
       comparable across d_head; the model's own training seed no longer leaks into eval.
  S3 — n_positions = seq_len is a *training* fact; measurement must cap its window to the
       model's n_positions (handled in the measure/ side, not here).
  Data — WikiText is tokenized INCREMENTALLY (the whole-corpus tokenize OOM'd at ~8.6GB).
"""

from __future__ import annotations

import math
import time
import statistics

import torch


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_wikitext103_tokens(tokenizer, split, max_tokens=None, doc_chunk=2000):
    """WikiText-103 split as a 1-D token tensor, tokenized incrementally (RAM-safe).

    Tokenizing the whole concatenated corpus at once OOM'd (~8.6GB for the train split).
    We accumulate non-empty docs into chunks, tokenize each, and stop at max_tokens.
    """
    from datasets import load_dataset
    ds = None
    for repo in ("Salesforce/wikitext", "wikitext"):
        try:
            ds = load_dataset(repo, "wikitext-103-raw-v1", split=split); break
        except Exception:
            continue
    if ds is None:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)

    pieces, buf, total = [], [], 0
    for t in ds["text"]:
        if not t.strip():
            continue
        buf.append(t)
        if len(buf) >= doc_chunk:
            ids = tokenizer("\n\n".join(buf), return_tensors="pt")["input_ids"].squeeze(0)
            pieces.append(ids); total += ids.numel(); buf = []
            if max_tokens is not None and total >= max_tokens:
                break
    if buf and (max_tokens is None or total < max_tokens):
        ids = tokenizer("\n\n".join(buf), return_tensors="pt")["input_ids"].squeeze(0)
        pieces.append(ids)
    out = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long)
    if max_tokens is not None:
        out = out[:max_tokens]
    return out


def _batch(ids, seq_len, batch_size, device, generator):
    n = ids.numel() - seq_len - 1
    starts = torch.randint(0, n, (batch_size,), generator=generator)
    x = torch.stack([ids[s:s + seq_len] for s in starts]).to(device)
    y = torch.stack([ids[s + 1:s + 1 + seq_len] for s in starts]).to(device)
    return x, y


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def build_model(model_cfg):
    """GPT-2 from a ModelCfg. Only n_head varies across the ablation."""
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(
        n_embd=model_cfg.d_model, n_layer=model_cfg.n_layer, n_head=model_cfg.n_head,
        n_positions=model_cfg.seq_len, n_ctx=model_cfg.seq_len,
        vocab_size=model_cfg.vocab_size,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1)
    return GPT2LMHeadModel(cfg)


# ---------------------------------------------------------------------------
# eval (S1: fixed eval seed, decoupled from the training seed)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_ids, seq_len, device, *, eval_seed=0, n_batches=40, batch_size=16):
    was_training = model.training
    model.eval()
    bs = min(batch_size, max(1, (val_ids.numel() - seq_len - 1) // seq_len))
    g = torch.Generator().manual_seed(eval_seed)         # FIXED across variants
    losses = []
    for _ in range(n_batches):
        x, y = _batch(val_ids, seq_len, bs, device, g)
        losses.append(model(x, labels=y).loss.item())
    if was_training:
        model.train()
    return statistics.mean(losses)


# ---------------------------------------------------------------------------
# training with snapshots (P5)
# ---------------------------------------------------------------------------

def train_with_snapshots(model_cfg, train_cfg, *, seed, device, train_ids, val_ids,
                         on_snapshot=None, verbose=True):
    """Train one variant; call on_snapshot(model, step, frac, val_loss) at each
    pre-registered snapshot step (P5). Returns (model, history).

    on_snapshot is where the experiment driver measures O_h / d_int — training itself
    stays measurement-free (audit: do not mix training and measurement).
    """
    torch.manual_seed(seed)
    model = build_model(model_cfg).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                            weight_decay=train_cfg.weight_decay, betas=(0.9, 0.95))
    steps, warmup, lr = train_cfg.steps, train_cfg.warmup, train_cfg.lr
    # fp16 autocast on cuda only (1.5-2x on T4). Does NOT change the experiment: the same
    # steps, batch, and data; only compute precision. The cross-arch gate already verified
    # the residual geometry is fp16-robust (GPT-2 fp16 O_h=0.283 vs fp32 0.284). CPU stays
    # fp32 so local smoke tests are unaffected.
    use_amp = str(device).startswith("cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)        # torch >= 2.4 API
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)           # older torch


    def lr_at(step):
        if step < warmup:
            return lr * step / max(1, warmup)
        prog = (step - warmup) / max(1, steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    snap_steps = set(train_cfg.snapshot_steps())
    g = torch.Generator().manual_seed(seed + 1)
    hist = {"step": [], "train_loss": [], "val_step": [], "val_loss": []}
    t0 = time.time()
    if verbose:
        print(f"  [train] {model_cfg.tag()} n_head={model_cfg.n_head} seed={seed} "
              f"steps={steps} snapshots={sorted(snap_steps)} device={device}")

    for step in range(1, steps + 1):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        x, y = _batch(train_ids, model_cfg.seq_len, train_cfg.batch_size, device, g)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = model(x, labels=y).loss
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step % train_cfg.log_every == 0:
            hist["step"].append(step); hist["train_loss"].append(loss.item())

        if step in snap_steps:
            vl = evaluate(model, val_ids, model_cfg.seq_len, device,
                          eval_seed=train_cfg.eval_seed, n_batches=train_cfg.eval_batches)
            hist["val_step"].append(step); hist["val_loss"].append(vl)
            frac = step / steps
            if verbose:
                print(f"    snapshot step {step:>5}/{steps} ({frac:>4.0%})  "
                      f"train {loss.item():.3f}  val {vl:.3f}  ({time.time()-t0:.0f}s)")
            if on_snapshot is not None:
                on_snapshot(model, step, frac, vl)
            model.train()

    return model, hist
