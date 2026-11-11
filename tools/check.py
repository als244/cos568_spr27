"""`make check`: verify the course setup on a GPU node and say what is wrong, if anything."""
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ok = True


def report(good: bool, what: str, fix: str = "", required: bool = True):
    """Print one line of the checklist, with a hint if the check failed."""
    global ok
    ok &= good or not required
    print(f"  [{'ok' if good else 'FAIL' if required else 'warn'}] {what}" + ("" if good else f"\n         -> {fix}"))


def locked(package: str) -> str:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    return next(p["version"] for p in lock["package"] if p["name"] == package)


def main():
    print("COS568 setup check")
    report("SLURM_JOB_ID" in os.environ, "running inside a Slurm job",
           "use `make check`, which submits itself to a GPU node")
    v = sys.version_info
    report((v.major, v.minor) == (3, 12), f"Python {v.major}.{v.minor}.{v.micro}",
           "this is not the course environment; run through `make` or `source setup/env.sh`")

    import torch
    want = locked("torch")
    report(torch.__version__.split("+")[0] == want, f"PyTorch {torch.__version__}",
           f"uv.lock pins torch {want}; ask the course staff to rebuild the environment")
    report(torch.cuda.is_available(), "CUDA is available", "no GPU visible; run inside a GPU job")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        gib = torch.cuda.get_device_properties(0).total_memory / 2**30
        report("L40" in name, f"GPU: {name}, {gib:.1f} GiB",
               "the assignments assume an NVIDIA L40; numbers will differ on this GPU", required=False)

    nsys = shutil.which("nsys")
    version = subprocess.run([nsys, "--version"], capture_output=True, text=True).stdout.strip() if nsys else ""
    report(bool(nsys), version or "Nsight Systems (nsys)", "nsys is not on PATH on this node")

    weights = Path(os.environ.get("OLMOE_WEIGHTS", ""))
    shards = sorted(weights.glob("*.safetensors")) if weights.is_dir() else []
    size = sum(f.stat().st_size for f in shards) / 1e9
    report((weights / "config.json").exists() and size > 13, f"OLMoE checkpoint: {weights} ({size:.1f} GB)",
           "checkpoint missing or incomplete; tell the course staff")

    home = Path(os.environ.get("COS568_HOME", ""))
    report(home.is_dir() and os.access(home, os.W_OK), f"your course directory: {home}",
           "it does not exist or is not writable; tell the course staff (needed for `make handin`)", required=False)

    print("All good." if ok else "Something is wrong; see the FAIL lines above.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
