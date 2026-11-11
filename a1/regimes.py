"""Experiment 2: what the GPU's own counters say about three kinds of work.

    python a1/regimes.py           # writes results/a1/regimes.csv

Runs each workload for a few seconds while sampling the GPU: how much of the
time a kernel was running, how much of the time memory was busy, the clock
frequency, and the power draw.
"""
import time

from common import require_gpu, write_csv

import torch

from olmoe.prof import GpuMonitor, warm_up

BF16 = dict(device="cuda", dtype=torch.bfloat16)


def workloads():
    """(name, function) for one workload of each kind."""
    a, b = torch.randn(8192, 8192, **BF16), torch.randn(8192, 8192, **BF16)
    yield "large GEMM, 8192^3", lambda: torch.matmul(a, b)
    x, y = torch.randn(2**18, 2048, **BF16), torch.randn(2**18, 2048, **BF16)
    yield "add two 1 GiB tensors", lambda: x + y
    rows, w = torch.randn(64, 2048, **BF16), torch.randn(2048, 1024, **BF16)
    yield "small GEMM, 64 x 2048 x 1024", lambda: torch.matmul(rows, w)


def main(seconds: float = 4.0):
    require_gpu()
    out = []
    for name, fn in workloads():
        warm_up(fn)
        calls, start = 0, time.time()
        with GpuMonitor() as mon:
            while time.time() - start < seconds:
                for _ in range(20):
                    fn()
                torch.cuda.synchronize()
                calls += 20
            ms = (time.time() - start) * 1e3 / calls
        out.append(dict(workload=name, ms_per_call=round(ms, 4), gpu_util=round(mon.gpu_util),
                        mem_util=round(mon.mem_util), clock_mhz=round(mon.clock_mhz), power_w=round(mon.power_w)))
        print(f"{name:28s} {out[-1]['ms_per_call']:8.4f} ms  kernel running {out[-1]['gpu_util']:3d}%  "
              f"memory busy {out[-1]['mem_util']:3d}%  {out[-1]['clock_mhz']} MHz  {out[-1]['power_w']} W")
        time.sleep(2)  # let the GPU cool down between workloads
    write_csv("regimes.csv", out)


if __name__ == "__main__":
    main()
