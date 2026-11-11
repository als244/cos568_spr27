"""Experiment 5: time every stage of a training step.

    python a1/stages.py --layers 4 --micro-batch-size 4 --seqlen 2048              # results/a1/stages.csv
    python a1/stages.py --layers 4 --micro-batch-size 4 --seqlen 2048 --recompute  # results/a1/stages_rc.csv
"""
import argparse

import time

from common import require_gpu, write_csv

import torch

from olmoe import OLMOE_1B_7B, build_model, random_batch
from olmoe.prof import GpuMonitor, warm_up
from olmoe.train import make_optimizer, phase_times, stage_names, stage_times, train_step
from perfmodel import mfu, optimizer_floor_ms, state_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--micro-batch-size", type=int, default=4, help="sequences in the step: its global batch is one micro batch")
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--recompute", action="store_true")
    args = p.parse_args()
    require_gpu()

    cfg = OLMOE_1B_7B.with_(n_layers=args.layers)
    model = build_model(cfg, recompute=args.recompute)  # random weights: the step time is within 3% of pretrained
    opt = make_optimizer(model)
    batch = random_batch([args.seqlen] * args.micro_batch_size, cfg.vocab_size)
    warm_up(lambda: train_step(model, opt, batch))
    torch.cuda.reset_peak_memory_stats()
    t = stage_times(model, opt, batch)
    peak = torch.cuda.max_memory_allocated()

    print(f"{'stage':12s} {'forward ms':>11s} {'backward ms':>12s} {'recompute ms':>13s}")
    rows = []
    for name in stage_names(model):
        row = dict(stage=name, forward_ms=t["forward"][name], backward_ms=t["backward"][name],
                   recompute_ms=t["recompute"][name], optimizer_ms="")
        rows.append(row)
        print(f"{name:12s} {row['forward_ms']:11.2f} {row['backward_ms']:12.2f} {row['recompute_ms']:13.2f}")
    phases = phase_times(t)
    rows.append(dict(stage="total", forward_ms=phases["forward"], backward_ms=phases["backward"],
                     recompute_ms=phases["recompute"], optimizer_ms=phases["optimizer"]))
    step_ms = sum(phases.values())
    tokens_per_s = args.micro_batch_size * args.seqlen / step_ms * 1e3
    print(f"{'total':12s} {phases['forward']:11.2f} {phases['backward']:12.2f} {phases['recompute']:13.2f}")
    print(f"optimizer {phases['optimizer']:.2f} ms   (floor {optimizer_floor_ms(cfg):.2f} ms)")
    print(f"step {step_ms:.1f} ms = {tokens_per_s:,.0f} tokens/s, MFU {mfu(cfg, args.seqlen, tokens_per_s):.1%}")
    print(f"peak memory {peak / 2**30:.2f} GiB   (weights + gradients + Adam: {state_bytes(cfg) / 2**30:.2f} GiB)")
    write_csv("stages_rc.csv" if args.recompute else "stages.csv", rows)

    # The GPU's own counters, over a few seconds of training steps.
    start = time.time()
    with GpuMonitor() as mon:
        while time.time() - start < 4:
            train_step(model, opt, batch)
            torch.cuda.synchronize()
    print(f"counters: a kernel was running {mon.gpu_util:.0f}% of the time, memory was busy {mon.mem_util:.0f}%, "
          f"clock {mon.clock_mhz:.0f} MHz, {mon.power_w:.0f} W, {mon.temp_c:.0f} C")


if __name__ == "__main__":
    main()
