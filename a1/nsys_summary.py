"""Summarize an Nsight recording by kind of NVTX range.

    python a1/nsys_summary.py results/a1/NAME_nvtx_gpu_proj_trace.csv reports/a1/NAME.sqlite

a1/nsys.sh runs this for you. The first file is the report
`nsys stats --report nvtx_gpu_proj_trace`: one row per NVTX range, with the
span of the kernels that were launched inside it (the range's "projection"
onto the GPU). The second is the recording, exported to SQLite.

Ranges that differ only in the layer number are added up. For each kind of
range the table gives, per recorded step:

    GPU ms      the projected time. Ranges nest, so rows overlap: "layers.*"
                contains "layers.*.moe", which contains "experts".
    instances   how many ranges of this kind were opened
    GPU ops     the kernels and memory operations launched inside

and, when the recording has GPU metrics (nsys --gpu-metrics-devices), the
mean of the GPU's own counters over the projected time: how busy the cores
(SMs) were, how busy the tensor cores were, and how much of the memory
bandwidth was in use, for reading and for writing.

One row is not a range. "(backward)" is what is left of each step outside
"forward", "optimizer" and the recomputations ("recompute.layers.*").
"""
import bisect
import csv
import re
import sqlite3
import sys
from collections import defaultdict
from itertools import accumulate
from pathlib import Path

# GPU counters to report: column name -> words to look for in the counter's name.
METRICS = {"sm_active": "sms active", "tensor_active": "tensor active",
           "dram_read": "dram read", "dram_write": "dram write"}
NOT_BACKWARD = ("forward", "optimizer", "recompute.layers.*")


class Counter:
    """The samples of one GPU counter, for quick sums over a time window."""

    def __init__(self, samples):  # [(timestamp, value)], sorted by timestamp
        self.times = [t for t, _ in samples]
        self.sums = [0, *accumulate(v for _, v in samples)]

    def window(self, start, end):
        """(sum, count) of the samples with start <= timestamp < end."""
        i, j = bisect.bisect_left(self.times, start), bisect.bisect_left(self.times, end)
        return self.sums[j] - self.sums[i], j - i


def load_ranges(trace: Path):
    """[(kind, projected start, projected end, GPU ops)] for every range instance."""
    ranges = []
    with open(trace) as f:
        for row in csv.DictReader(f):
            start = int(row["Projected Start (ns)"])
            kind = re.sub(r"layers\.\d+", "layers.*", row["Name"].lstrip(":"))
            ranges.append((kind, start, start + int(row["Projected Duration (ns)"]), int(row["NumGPUOps"])))
    return ranges


def load_counters(database):
    """{column: Counter} for the GPU counters that the recording has; {} if it has none."""
    tables = {name for (name,) in database.execute("select name from sqlite_master where type = 'table'")}
    if not {"GPU_METRICS", "TARGET_INFO_GPU_METRICS"} <= tables:
        return {}
    recorded = database.execute("select typeId, metricId, metricName from TARGET_INFO_GPU_METRICS").fetchall()
    counters = {}
    for column, words in METRICS.items():
        match = next(((t, m) for t, m, name in recorded if words in name.lower()), None)
        if match:
            samples = database.execute("select timestamp, value from GPU_METRICS where typeId = ? and metricId = ? "
                                       "order by timestamp", match).fetchall()
            counters[column] = Counter(samples)
    return counters


def describe_gpu(database) -> str:
    """The GPU as its driver describes it: Nsight stores that in every recording."""
    try:
        name, sms, clock, bandwidth, l2 = database.execute(
            "select name, smCount, clockRate, memoryBandwidth, l2CacheSize from TARGET_INFO_GPU").fetchone()
        return (f"{name}: {sms} SMs, {clock / 1e6:.0f} MHz, memory bandwidth {bandwidth / 1e9:.0f} GB/s, "
                f"L2 cache {l2 / 2**20:.0f} MiB")
    except (sqlite3.Error, TypeError):
        return ""


def main(trace: Path, sqlite_file: Path = None):
    """Print the table and write it next to the trace, as NAME_ranges.csv."""
    ranges = load_ranges(trace)
    database = sqlite3.connect(sqlite_file) if sqlite_file and sqlite_file.exists() else None
    counters = load_counters(database) if database else {}

    windows = defaultdict(list)  # kind -> [(start, end)]
    ops = defaultdict(int)
    for kind, start, end, n_ops in ranges:
        windows[kind].append((start, end))
        ops[kind] += n_ops
    steps = windows.get("step") or [(min(s for _, s, _, _ in ranges), max(e for _, _, e, _ in ranges))]

    def measure(include, exclude=()):
        """GPU ms, and the mean of each counter, over the `include` windows minus the `exclude` windows."""
        ns = sum(e - s for s, e in include) - sum(e - s for s, e in exclude)
        means = {}
        for column, counter in counters.items():
            total = count = 0
            for sign, group in ((1, include), (-1, exclude)):
                for start, end in group:
                    window_sum, window_count = counter.window(start, end)
                    total, count = total + sign * window_sum, count + sign * window_count
            means[column] = round(total / count, 1) if count > 0 else ""
        return ns / len(steps) / 1e6, means

    rows = []
    for kind in windows:
        ms, means = measure(windows[kind])
        rows.append(dict(range=kind, gpu_ms_per_step=round(ms, 3), instances_per_step=len(windows[kind]) / len(steps),
                         gpu_ops_per_step=ops[kind] / len(steps), **means))
    if "optimizer" in windows:  # a training step: the backward pass is whatever is left of it
        ms, means = measure(steps, [w for kind in NOT_BACKWARD for w in windows.get(kind, [])])
        rows.append(dict(range="(backward)", gpu_ms_per_step=round(ms, 3), instances_per_step=1.0,
                         gpu_ops_per_step="", **means))
    rows.sort(key=lambda row: -row["gpu_ms_per_step"])

    if database and describe_gpu(database):
        print(describe_gpu(database))
    print(f"{'range':34s} {'GPU ms/step':>11s} {'instances':>9s} {'GPU ops':>8s}"
          + "".join(f" {column.replace('_', ' ') + ' %':>15s}" for column in counters))
    for row in rows:
        print(f"{row['range'][:34]:34s} {row['gpu_ms_per_step']:11.2f} {row['instances_per_step']:9g} "
              f"{row['gpu_ops_per_step']:>8}" + "".join(f" {row[column]:>15}" for column in counters))
    if not counters:
        print("(no GPU metrics in this recording)")

    out = trace.with_name(trace.name.replace("_nvtx_gpu_proj_trace.csv", "_ranges.csv"))
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*(Path(arg) for arg in sys.argv[1:3]))
