"""Experiment 4: one global batch, run as micro batches of different sizes.

    python a1/micro_batches.py     # writes results/a1/micro_batches.csv and micro_batches.png
    python a1/micro_batches.py --layers 2 --global-batch-size 8 --seqlen 1024 --micro-batch-sizes 8 4 1

The global batch of a step is `--global-batch-size` sequences of `--seqlen`
tokens. The script runs that step in micro batches of each of the given
sizes (your accumulated_step), without and with recomputation, and reports
the time of a step, the throughput, and the peak memory.

It also draws memory over the course of one step, for a global batch of half
that size (so that it still fits as a single micro batch), run in four ways:
as one micro batch or as four, without or with recomputation.
"""
import argparse

from common import require_gpu, save_plot, write_csv

import torch

from olmoe import OLMOE_1B_7B, build_model, random_batch
from olmoe.prof import cuda_time, median, memory_timeline, warm_up
from olmoe.train import accumulated_step, make_optimizer


def micro_batches(cfg, global_batch_size: int, micro_batch_size: int, seqlen: int):
    """A global batch of random sequences, as a list of micro batches."""
    return [random_batch([seqlen] * micro_batch_size, cfg.vocab_size, seed=i)
            for i in range(global_batch_size // micro_batch_size)]


def measure(model, opt, batches):
    """Time one step on these micro batches. Returns (step ms, peak GiB), or None if it does not fit in memory."""
    def step():
        accumulated_step(model, opt, batches)

    try:
        torch.cuda.empty_cache()
        warm_up(step)
        torch.cuda.reset_peak_memory_stats()
        ms = median(cuda_time(step, warmup=0, iters=5))
        return ms, torch.cuda.max_memory_allocated() / 2**30
    except torch.OutOfMemoryError:
        opt.zero_grad(set_to_none=True)
        return None


def plot(curves, cfg, seqlen: int, weights_gib: float):
    """curves: {(recompute, micro batch size): (ms, GiB), or None if the step did not fit}.
    Draws memory over one step, one panel for each."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes = sorted({size for _, size in curves}, reverse=True)
    weights_and_adam, with_gradients = 3 * weights_gib, 4 * weights_gib  # AdamW keeps two moments per weight
    fig, axes = plt.subplots(2, len(sizes), figsize=(4.6 * len(sizes), 5.6), sharey=True, squeeze=False)
    for row, recompute in enumerate((False, True)):
        for col, size in enumerate(sizes):
            ax, name = axes[row][col], f"micro batch size {size}, {'with' if recompute else 'no'} recomputation"
            for level in (weights_and_adam, with_gradients):
                ax.axhline(level, color="#52514e", linewidth=0.8, linestyle=":")
            if curves[recompute, size] is None:
                ax.set_title(f"{name}: out of memory", fontsize=9, loc="left")
            else:
                ms, gib = curves[recompute, size]
                ax.fill_between(ms, gib, step="post", color="#2a78d6", alpha=0.2, linewidth=0)
                ax.plot(ms, gib, drawstyle="steps-post", color="#2a78d6", linewidth=1.5)
                ax.set_title(f"{name}: peak {max(gib):.1f} GiB", fontsize=9, loc="left")
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("time since the step began (ms, on the CPU)")
    for ax in axes[:, 0]:
        ax.set_ylabel("allocated (GiB)")
    fig.suptitle(f"Memory over one step: global batch size {max(sizes)}, {seqlen} tokens per sequence, "
                 f"{cfg.n_layers} layers", fontsize=10, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Dotted lines: the weights with AdamW's two moments (lower), and the same plus one gradient "
             "per weight (upper).", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save_plot(fig, "micro_batches.png")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--global-batch-size", type=int, default=16, help="sequences per optimizer step")
    p.add_argument("--seqlen", type=int, default=2048, help="tokens per sequence")
    p.add_argument("--micro-batch-sizes", type=int, nargs="+", default=[16, 8, 4, 2, 1],
                   help="sequences per forward and backward pass; each must divide the global batch size")
    args = p.parse_args()
    require_gpu()

    cfg = OLMOE_1B_7B.with_(n_layers=args.layers)
    tokens = args.global_batch_size * args.seqlen
    print(f"global batch size {args.global_batch_size}, {args.seqlen} tokens per sequence: "
          f"{tokens:,} tokens per step, {args.layers} layers")
    print(f"{'micro batch size':>16s} {'micro batches':>14s} {'recompute':>10s} {'step ms':>9s} {'tokens/s':>9s} {'peak GiB':>9s}")
    rows, curves = [], {}
    for recompute in (False, True):
        model = build_model(cfg, recompute=recompute)  # random weights, as everywhere in Part 3
        opt = make_optimizer(model)
        weights_gib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30
        for size in args.micro_batch_sizes:
            if args.global_batch_size % size:
                continue
            result = measure(model, opt, micro_batches(cfg, args.global_batch_size, size, args.seqlen))
            row = dict(global_batch_size=args.global_batch_size, micro_batch_size=size, seqlen=args.seqlen,
                       n_layers=args.layers, recompute=int(recompute), step_ms="", tokens_per_s="", peak_gib="")
            line = f"{size:16d} {args.global_batch_size // size:14d} {'yes' if recompute else 'no':>10s}"
            if result is None:
                print(f"{line}   out of memory")
            else:
                ms, gib = result
                row.update(step_ms=round(ms, 1), tokens_per_s=round(tokens / ms * 1e3), peak_gib=round(gib, 2))
                print(f"{line} {ms:9.1f} {row['tokens_per_s']:9,d} {gib:9.2f}")
            rows.append(row)
        half = max(4, args.global_batch_size // 2)  # the figure: half the global batch, as one micro batch and as four
        for size in (half, half // 4):
            batches = micro_batches(cfg, half, size, args.seqlen)

            def step():
                accumulated_step(model, opt, batches)

            try:
                torch.cuda.empty_cache()
                warm_up(step)
                curves[recompute, size] = memory_timeline(step)
            except torch.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                curves[recompute, size] = None
        del model, opt
        torch.cuda.empty_cache()
    write_csv("micro_batches.csv", rows)
    plot(curves, cfg, args.seqlen, weights_gib)


if __name__ == "__main__":
    main()
