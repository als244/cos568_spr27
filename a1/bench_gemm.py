"""Experiment 2: measure matrix multiplies and place them on the L40 roofline.

    python a1/bench_gemm.py        # writes results/a1/gemm.csv, roofline.png, gemm_sustained.csv

For each GEMM shape that appears in OLMoE, sweeps M, the number of rows
(tokens), and times  C = torch.matmul(A, B)  in bf16, for A: [M, k] and B: [k, n]. Then runs one
large GEMM back to back for a few seconds, starting from an idle GPU.
"""
import time

from common import require_gpu, save_plot, steady_ms, write_csv

import torch

from olmoe.config import OLMOE_1B_7B as cfg
from olmoe.prof import GpuMonitor, cuda_time, median
from perfmodel import PEAK_GBPS, PEAK_TFLOPS, gemm_bytes, gemm_flops, roofline_tflops

H, I, V = cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size
SHAPES = {  # name: (n, k)
    "attention q_proj": (H, H),
    "expert gate_proj": (I, H),
    "expert down_proj": (H, I),
    "lm_head": (V, H),
    "square, n = k = 8192": (8192, 8192),
}
ROWS = [2**i for i in range(0, 16)]


def main():
    require_gpu()
    rows = []
    for name, (n, k) in SHAPES.items():
        b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
        for m in ROWS:
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ms = steady_ms(lambda: torch.matmul(a, b))
            flops, nbytes = gemm_flops(m, n, k), gemm_bytes(m, n, k)
            rows.append(dict(shape=name, m=m, n=n, k=k, ms=ms,
                             intensity=flops / nbytes,
                             tflops=flops / ms / 1e9,
                             gbps=nbytes / ms / 1e6,
                             roofline_tflops=roofline_tflops(flops / nbytes)))
            print(f"{name:22s} M={m:6d}  {ms:9.4f} ms  {rows[-1]['tflops']:7.2f} TFLOP/s  "
                  f"{rows[-1]['gbps']:7.1f} GB/s  intensity {flops / nbytes:7.1f}")
    write_csv("gemm.csv", rows)
    plot(rows)
    sustained()


def sustained(n: int = 8192, seconds: float = 60.0):
    """Start from an idle GPU and run one large GEMM back to back for a minute, watching the GPU's
    clock, power and temperature."""
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    torch.matmul(a, b)  # once, untimed: the result's memory is then in PyTorch's cache
    torch.cuda.synchronize()
    time.sleep(5)  # then let the GPU idle
    rows, start, shown = [], time.time(), -5.0
    while time.time() - start < seconds:
        with GpuMonitor(interval=0.02) as mon:
            ms = median(cuda_time(lambda: torch.matmul(a, b), warmup=0, iters=8))
        rows.append(dict(seconds=round(time.time() - start, 2), tflops=round(gemm_flops(n, n, n) / ms / 1e9, 1),
                         clock_mhz=round(mon.clock_mhz), power_w=round(mon.power_w), temp_c=round(mon.temp_c)))
        if rows[-1]["seconds"] < 1 or rows[-1]["seconds"] - shown >= 5:  # the first second in full, then every 5 s
            shown = rows[-1]["seconds"]
            print(f"t = {rows[-1]['seconds']:5.2f} s  {rows[-1]['tflops']:6.1f} TFLOP/s  clock {rows[-1]['clock_mhz']} MHz  "
                  f"power {rows[-1]['power_w']} W  {rows[-1]['temp_c']} C")
    write_csv("gemm_sustained.csv", rows)


def plot(rows):
    """Measured TFLOP/s against arithmetic intensity, over the roofline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [2.0**i for i in range(-2, 14)]
    ax.plot(xs, [roofline_tflops(x) for x in xs], "k-", lw=1, label="roofline")
    for name in SHAPES:
        pts = [r for r in rows if r["shape"] == name]
        ax.plot([r["intensity"] for r in pts], [r["tflops"] for r in pts], "o-", ms=3, label=name)
    ax.set(xscale="log", yscale="log", xlabel="arithmetic intensity (FLOPs per byte)",
           ylabel="TFLOP/s", title=f"bf16 GEMM on an L40 ({PEAK_TFLOPS:.0f} TFLOP/s, {PEAK_GBPS:.0f} GB/s)")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_plot(fig, "roofline.png")


if __name__ == "__main__":
    main()
