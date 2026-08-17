#!/usr/bin/env python3
"""Train a Transformer language model on a tokenized uint16 dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from cs336_basics.nn import TransformerLM
from cs336_basics.training import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True, help="Tokenized train .npy")
    parser.add_argument("--valid-data", type=Path, default=None, help="Tokenized valid .npy")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from")

    # Model (TinyStories defaults from CS336 handout-style configs)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    # Optimization
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=5_000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--cosine-cycle-iters", type=int, default=5_000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Logging / eval
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--ckpt-interval", type=int, default=500)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="cs336-a1")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(
    model: TransformerLM,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r") if args.valid_data else None
    print(f"train tokens: {len(train_data):,} | device: {device}")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at iteration {start_iter}")

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    model.train()
    t0 = time.time()
    for it in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if (it + 1) % args.log_interval == 0:
            tokens_per_sec = (
                args.log_interval * args.batch_size * args.context_length / max(time.time() - t0, 1e-8)
            )
            print(f"iter {it + 1:6d} | loss {loss.item():.4f} | lr {lr:.6e} | tok/s {tokens_per_sec:,.0f}")
            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item(), "train/lr": lr, "train/tok_s": tokens_per_sec}, step=it + 1)
            t0 = time.time()

        if valid_data is not None and (it + 1) % args.eval_interval == 0:
            val_loss = estimate_loss(
                model,
                valid_data,
                args.batch_size,
                args.context_length,
                device,
                args.eval_batches,
            )
            print(f"iter {it + 1:6d} | valid loss {val_loss:.4f}")
            if wandb_run is not None:
                wandb_run.log({"valid/loss": val_loss}, step=it + 1)

        if (it + 1) % args.ckpt_interval == 0 or (it + 1) == args.max_iters:
            ckpt_path = args.out_dir / f"ckpt_{it + 1:06d}.pt"
            save_checkpoint(model, optimizer, it + 1, ckpt_path)
            save_checkpoint(model, optimizer, it + 1, args.out_dir / "ckpt_latest.pt")
            print(f"Saved checkpoint -> {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
