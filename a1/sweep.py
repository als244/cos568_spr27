"""Experiment 7: step time and its phases, throughput, peak memory and GPU counters, over shapes of the step.

    python a1/sweep.py                         # results/a1/sweep.csv
    python a1/sweep.py --recompute --layers 2 4 8 --micro-batch-sizes 2 8 --seqlens 1024 4096

Rows are appended to results/a1/sweep.csv, so several runs accumulate into one
table (delete the file to start over). This is the data your predictor in
Coding Exercise 9 is calibrated on: choose the configurations you need.
Use `sbatch slurm/job.sbatch python a1/sweep.py ...` for long sweeps.

A step should take at least 100 ms: with four layers, that means at least
2,048 tokens. Shorter steps are marked with a star, because their times move
by several percent from one run to the next, whatever the warm-up.

No configuration is measured on a cold chip. If the chip is below 60 degrees,
the GPU is first kept busy until it gets there (heat_up in olmoe/prof.py):
about 40 seconds from an idle GPU, and nothing once a sweep is under way.
"""
import argparse
import csv
import time

from common import RESULTS, ROOT, require_gpu

import torch

from olmoe import OLMOE_1B_7B, build_model, random_batch
from olmoe.prof import GpuMonitor, cuda_time, heat_up, median, warm_up
from olmoe.train import make_optimizer, phase_times, stage_times, train_step
from perfmodel import mfu

COLUMNS = ["n_layers", "micro_batch_size", "seqlen", "recompute", "step_ms", "forward_ms", "backward_ms", "recompute_ms",
           "optimizer_ms", "tokens_per_s", "mfu", "peak_gib", "gpu_util", "mem_util", "clock_mhz", "temp_c"]
SHORT_MS = 100  # a step shorter than this is not timed reliably


def measure(model, opt, cfg, micro_batch_size, seqlen, recompute):
    """Measure one configuration. Returns its row of the table."""
    data = random_batch([seqlen] * micro_batch_size, cfg.vocab_size)  # T = micro_batch_size * seqlen tokens
    def step():
        train_step(model, opt, data)

    waited = heat_up()  # a hot chip is slower: compare like with like
    if waited > 1:
        print(f"(heated the chip to {torch.cuda.temperature()} C first: {waited:.0f} s)")
    torch.cuda.empty_cache()
    warm_up(step)
    torch.cuda.reset_peak_memory_stats()
    step_ms = median(cuda_time(step, warmup=0, iters=5))
    temp_c = torch.cuda.temperature()  # of the chip, as these steps ended
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    phases = phase_times(stage_times(model, opt, data, warmup=1, iters=5))
    start = time.time()
    with GpuMonitor() as counters:  # the GPU's own view of a few seconds of these steps
        while time.time() - start < 2.5:
            step()
            torch.cuda.synchronize()
    tokens_per_s = micro_batch_size * seqlen / step_ms * 1e3
    return dict(n_layers=cfg.n_layers, micro_batch_size=micro_batch_size, seqlen=seqlen, recompute=int(recompute),
                step_ms=round(step_ms, 3), forward_ms=round(phases["forward"], 3),
                backward_ms=round(phases["backward"], 3), recompute_ms=round(phases["recompute"], 3),
                optimizer_ms=round(phases["optimizer"], 3),
                tokens_per_s=round(tokens_per_s, 1), mfu=round(mfu(cfg, seqlen, tokens_per_s), 4),
                peak_gib=round(peak_gib, 3), gpu_util=round(counters.gpu_util),
                mem_util=round(counters.mem_util), clock_mhz=round(counters.clock_mhz), temp_c=temp_c)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", type=int, nargs="+", default=[4])
    p.add_argument("--micro-batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16],
                   help="sequences per step: every step here is one micro batch")
    p.add_argument("--seqlens", type=int, nargs="+", default=[1024, 2048, 4096])
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()
    require_gpu()

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "sweep.csv"
    new = not path.exists()
    short = False
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            writer.writeheader()
        for n_layers in args.layers:
            cfg = OLMOE_1B_7B.with_(n_layers=n_layers)
            model = build_model(cfg, recompute=args.recompute)  # random weights: speed barely depends on them
            opt = make_optimizer(model)
            for seqlen in args.seqlens:
                for size in args.micro_batch_sizes:
                    try:
                        row = measure(model, opt, cfg, size, seqlen, args.recompute)
                    except torch.OutOfMemoryError:
                        opt.zero_grad(set_to_none=True)
                        print(f"{n_layers:2d} layers {size:3d} x {seqlen:<5d}  out of memory")
                        continue
                    writer.writerow(row)
                    f.flush()
                    star = " *" if row["step_ms"] < SHORT_MS else ""
                    short = short or bool(star)
                    print(f"{n_layers:2d} layers {size:3d} x {seqlen:<5d}{row['step_ms']:7.1f} ms {row['tokens_per_s']:7,.0f} tok/s  "
                          f"MFU {row['mfu']:5.1%} {row['peak_gib']:5.1f} GiB  kernel {row['gpu_util']:3d}%  "
                          f"memory {row['mem_util']:3d}%  {row['clock_mhz']} MHz  {row['temp_c']} C{star}")
            del model, opt
            torch.cuda.empty_cache()
    if short:
        print(f"* shorter than {SHORT_MS} ms: too short a step to time reliably. Give it more tokens.")
    print(f"appended to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
