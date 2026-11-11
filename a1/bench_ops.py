"""Experiment 2: memory-bound operations. Achieved bandwidth against tensor size.

    python a1/bench_ops.py         # writes results/a1/ops.csv and ops.png

"Bytes" is the least traffic each operation needs: every input read once and
the output written once. Bandwidth = bytes / time.
"""
from common import require_gpu, save_plot, steady_ms, write_csv

import torch
import torch.nn.functional as F
from perfmodel import L2_BYTES, PEAK_GBPS

H = 2048
OPS = {  # name: (function of two [rows, H] bf16 tensors, tensors read + tensors written)
    "copy": (lambda x, y: x.clone(), 2),
    "add": (lambda x, y: x + y, 3),
    "silu(x) * y": (lambda x, y: F.silu(x) * y, 3),
    "rms_norm": (lambda x, y: F.rms_norm(x, (H,)), 2),
}


def main():
    require_gpu()
    rows = []
    for name, (fn, tensors) in OPS.items():
        for n in [2**i for i in range(4, 19)]:
            x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16)
            y = torch.randn_like(x)
            ms = steady_ms(lambda: fn(x, y))
            nbytes = tensors * n * H * 2
            rows.append(dict(op=name, rows=n, mib=nbytes / 2**20, ms=ms, gbps=nbytes / ms / 1e6))
            print(f"{name:12s} {nbytes / 2**20:9.2f} MiB  {ms:8.4f} ms  {rows[-1]['gbps']:7.1f} GB/s")
    write_csv("ops.csv", rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in OPS:
        pts = [r for r in rows if r["op"] == name]
        ax.plot([r["mib"] for r in pts], [r["gbps"] for r in pts], "o-", ms=3, label=name)
    ax.axhline(PEAK_GBPS, color="k", lw=1, ls="--", label="memory bandwidth")
    ax.axvline(L2_BYTES / 2**20, color="gray", lw=1, ls=":", label="L2 cache size")
    ax.set(xscale="log", xlabel="bytes moved (MiB)", ylabel="achieved GB/s")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    save_plot(fig, "ops.png")


if __name__ == "__main__":
    main()
