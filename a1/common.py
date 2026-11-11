"""Shared helpers for the a1 scripts."""
import csv
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make `import olmoe` work however the script is started
RESULTS = ROOT / "results" / "a1"
REPORTS = ROOT / "reports" / "a1"


def require_gpu():
    """Stop with a helpful message unless there is a GPU to use. On a cluster
    that means being inside a Slurm job, never on the login node."""
    if shutil.which("srun") and "SLURM_JOB_ID" not in os.environ:
        sys.exit("Not inside a Slurm job. GPU work must not run on the login node:\n"
                 "run this inside `make gpu-shell` or with `sbatch slurm/job.sbatch`.")
    import torch
    if not torch.cuda.is_available():
        sys.exit("No GPU here.")


def write_csv(name: str, rows: list[dict]) -> Path:
    """Write a list of dicts as results/a1/<name>, one row each."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def save_plot(fig, name: str):
    """Save a matplotlib figure as results/a1/<name>."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RESULTS / name, dpi=150)
    print(f"wrote {(RESULTS / name).relative_to(ROOT)}")


def steady_ms(fn) -> float:
    """Median GPU time (ms) of fn() in steady state: after a warm-up, and over
    enough calls to fill about 200 ms (at least 5, at most 2000)."""
    from olmoe.prof import cuda_time, median, warm_up
    warm_up(fn)
    once = median(cuda_time(fn, warmup=0, iters=5))
    return median(cuda_time(fn, warmup=0, iters=min(2000, max(5, int(200 / once)))))
