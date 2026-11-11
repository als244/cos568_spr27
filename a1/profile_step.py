"""Run a few forward passes or training steps for a profiler to watch.

Use it through a1/nsys.sh, which adds the Nsight Systems flags:

    a1/nsys.sh fwd_small  --mode forward --micro-batch-size 1 --seqlen 512
    a1/nsys.sh train      --mode train --layers 4 --micro-batch-size 4 --seqlen 2048

Only the `--steps` measured iterations are recorded. The warm-up runs outside
the capture range, so the report stays small and shows steady-state behavior.
"""
import argparse

from common import REPORTS, require_gpu

import torch

from olmoe import OLMOE_1B_7B, build_model, load_model, random_batch
from olmoe.prof import annotate, cuda_time, median, memory_snapshot, nvtx, warm_up
from olmoe.train import make_optimizer, train_step


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["forward", "train"], default="forward")
    p.add_argument("--micro-batch-size", type=int, default=1,
                   help="sequences per pass; a training step's global batch is one micro batch")
    p.add_argument("--seqlen", type=int, default=512, help="tokens per sequence")
    p.add_argument("--layers", type=int, default=None, help="keep only the first N layers (default: all 16)")
    p.add_argument("--random-init", action="store_true", help="random weights instead of the checkpoint")
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--moe-impl", choices=["grouped", "naive"], default="grouped")
    p.add_argument("--warmup", type=int, default=3, help="untimed steps before measuring (at least this many)")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--nvtx-depth", type=int, default=3)
    p.add_argument("--memory-snapshot", metavar="NAME", help="also write reports/a1/NAME.pickle")
    args = p.parse_args()
    require_gpu()

    kw = dict(recompute=args.recompute, moe_impl=args.moe_impl)
    if args.random_init:
        model = build_model(OLMOE_1B_7B.with_(n_layers=args.layers or OLMOE_1B_7B.n_layers), **kw)
    else:
        model = load_model(n_layers=args.layers, **kw)
    annotate(model, depth=args.nvtx_depth)
    batch = random_batch([args.seqlen] * args.micro_batch_size, model.cfg.vocab_size)

    if args.mode == "forward":
        model.eval()

        @torch.no_grad()
        def step():
            with nvtx("forward"):
                model(batch.tokens, batch.seqs)
    else:
        opt = make_optimizer(model)

        def step():
            train_step(model, opt, batch)

    warm_up(step, calls=args.warmup)
    before = median(cuda_time(step, warmup=0, iters=5))
    torch.cuda.reset_peak_memory_stats()

    def recorded_step():
        with nvtx("step"):
            step()

    torch.cuda.profiler.start()
    recorded = median(cuda_time(recorded_step, warmup=0, iters=args.steps))
    torch.cuda.profiler.stop()
    print(f"one step: {before:.1f} ms before the recording started, {recorded:.1f} ms while recording")
    print(f"peak memory allocated: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

    if args.memory_snapshot:
        # One more step, outside the capture range: recording allocations slows the CPU down.
        REPORTS.mkdir(parents=True, exist_ok=True)
        with memory_snapshot(str(REPORTS / f"{args.memory_snapshot}.pickle")):
            step()
            torch.cuda.synchronize()
        print(f"wrote reports/a1/{args.memory_snapshot}.pickle")


if __name__ == "__main__":
    main()
