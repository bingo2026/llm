#!/usr/bin/env python3
"""End-to-end benchmarking for CS336 Assignment 2 (problem: benchmarking_script).

Times forward / forward+backward / full training steps of a Transformer LM,
with optional BF16 autocast and CUDA memory snapshots.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path
from timeit import default_timer as timer

import torch

# Allow running without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "assignment1-basics"))
sys.path.insert(0, str(ROOT))

from cs336_basics.nn import TransformerLM  # noqa: E402
from cs336_basics.training import AdamW, cross_entropy  # noqa: E402


try:
    import torch.cuda.nvtx as nvtx
except Exception:  # pragma: no cover
    nvtx = None


def nvtx_range(name: str):
    """NVTX range for Nsight; no-op on CPU / builds without NVTX."""
    if nvtx is None or not torch.cuda.is_available():
        return nullcontext()
    try:
        return nvtx.range(name)
    except RuntimeError:
        return nullcontext()


MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Preset from handout Table 1. Overrides individual size args.",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--d-ff", type=int, default=3072)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up steps before timing (w).")
    parser.add_argument("--steps", type=int, default=10, help="Timed measurement steps (n).")
    parser.add_argument(
        "--mode",
        type=str,
        default="training",
        choices=["forward", "forward_backward", "training", "all"],
        help="What to time. 'all' runs forward, forward_backward, and training, then reports a breakdown.",
    )

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", help="Use torch.autocast with bfloat16.")
    parser.add_argument(
        "--memory-snapshot",
        type=Path,
        default=None,
        help="If set, dump a CUDA memory snapshot pickle after timed steps.",
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to write timings JSON.")
    return parser.parse_args()


def resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def build_model(args: argparse.Namespace, device: torch.device) -> TransformerLM:
    cfg = dict(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    if args.model_size is not None:
        cfg.update(MODEL_CONFIGS[args.model_size])

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=args.rope_theta,
        device=device,
    )
    return model.to(device)


def make_batch(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
    return x, y


def run_step(
    model: TransformerLM,
    optimizer: AdamW | None,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str,
    autocast_ctx,
) -> None:
    if mode == "forward":
        with nvtx_range("forward"), autocast_ctx:
            _ = model(x)
        return

    optimizer.zero_grad(set_to_none=True)
    with nvtx_range("forward"), autocast_ctx:
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    with nvtx_range("backward"):
        loss.backward()
    if mode == "training":
        with nvtx_range("optimizer"):
            assert optimizer is not None
            optimizer.step()


def benchmark(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    model = build_model(args, device)
    model.train(args.mode != "forward")

    optimizer = None
    if args.mode in {"forward_backward", "training"}:
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    x, y = make_batch(args, device)

    if args.bf16:
        if device.type != "cuda":
            raise SystemExit("--bf16 requires a CUDA device")
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext()

    # Warm-up (excluded from NVTX capture via label)
    with nvtx_range("warmup"):
        for _ in range(args.warmup):
            run_step(model, optimizer, x, y, args.mode, autocast_ctx)
            synchronize(device)

    if args.memory_snapshot is not None:
        if device.type != "cuda":
            raise SystemExit("--memory-snapshot requires a CUDA device")
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)

    times: list[float] = []
    with nvtx_range("timed_steps"):
        for _ in range(args.steps):
            synchronize(device)
            t0 = timer()
            run_step(model, optimizer, x, y, args.mode, autocast_ctx)
            synchronize(device)
            times.append(timer() - t0)

    if args.memory_snapshot is not None:
        args.memory_snapshot.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(args.memory_snapshot))
        torch.cuda.memory._record_memory_history(enabled=None)

    mean_s = statistics.mean(times)
    std_s = statistics.stdev(times) if len(times) > 1 else 0.0
    result = {
        "model_size": args.model_size,
        "mode": args.mode,
        "device": str(device),
        "bf16": args.bf16,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "vocab_size": args.vocab_size,
        "d_model": model.token_embeddings.weight.shape[1],
        "num_layers": len(model.layers),
        "num_heads": model.layers[0].attn.num_heads,
        "d_ff": model.layers[0].ffn.w1.weight.shape[0],
        "warmup": args.warmup,
        "steps": args.steps,
        "times_s": times,
        "mean_s": mean_s,
        "std_s": std_s,
        "mean_ms": mean_s * 1000.0,
        "std_ms": std_s * 1000.0,
    }
    return result


def main() -> None:
    args = parse_args()
    modes = ["forward", "forward_backward", "training"] if args.mode == "all" else [args.mode]
    results = []
    for mode in modes:
        run_args = argparse.Namespace(**vars(args))
        run_args.mode = mode
        # Only dump memory on the last mode when sweeping.
        if args.mode == "all" and mode != modes[-1]:
            run_args.memory_snapshot = None
        result = benchmark(run_args)
        results.append(result)
        print(
            f"mode={result['mode']:<18} model={result['model_size'] or 'custom'} "
            f"device={result['device']} bf16={result['bf16']}  "
            f"mean={result['mean_ms']:.3f} ms  std={result['std_ms']:.3f} ms"
        )

    if len(results) == 3:
        fwd, fb, full = (r["mean_ms"] for r in results)
        print(
            f"breakdown (approx): forward={fwd:.3f} ms | "
            f"backward≈{fb - fwd:.3f} ms | optimizer≈{full - fb:.3f} ms"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = results if len(results) > 1 else results[0]
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()
