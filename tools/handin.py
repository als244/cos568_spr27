"""`make handin`: check your work and copy it to your course directory.

You may hand in as often as you like. The staff grade your last hand-in
before the deadline, by running the public and hidden tests on it.
"""
import getpass
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Per assignment: the files the staff grade, and the generated files that must come along.
GRADED = {
    "1": ["olmoe/prof.py", "olmoe/train.py", "olmoe/model.py", "a1/perfmodel.py", "a1/answers.md"],
}
REQUIRED = {
    "1": [f"results/a1/{f}" for f in (
              "gemm.csv", "roofline.png", "gemm_sustained.csv", "ops.csv", "ops.png", "regimes.csv",
              "fwd_small_ranges.csv", "fwd_large_ranges.csv", "fwd_small_naive_ranges.csv",
              "micro_batches.csv", "micro_batches.png", "stages.csv", "stages_rc.csv",
              "train_ranges.csv", "train_rc_ranges.csv", "train_naive_ranges.csv", "sweep.csv")]
         + [f"reports/a1/{f}" for f in (
              "fwd_small.nsys-rep", "fwd_large.nsys-rep", "fwd_small_naive.nsys-rep",
              "train.nsys-rep", "train_rc.nsys-rep", "train_naive.nsys-rep", "train.pickle", "train_rc.pickle")],
}
EXTRA = {"1": ["results/a1/*.csv", "results/a1/*.png", "reports/a1/*.nsys-rep", "reports/a1/*.pickle"]}
MAX_MB = 500


def problems(a: str) -> list[str]:
    """What is still missing from assignment `a`: files, and unanswered questions."""
    found = [f"missing {f}" for f in GRADED[a] + REQUIRED[a] if not (ROOT / f).exists()]
    answers = ROOT / f"a{a}" / "answers.md"
    if answers.exists():
        blank = answers.read_text().count("_(your answer)_")
        if blank:
            found.append(f"a{a}/answers.md: {blank} answer(s) still say _(your answer)_")
    return found


def main(a: str):
    found = problems(a)
    if found:
        print("Not complete yet:\n  " + "\n  ".join(found))
        if os.environ.get("FORCE") != "1" and input("Hand in anyway? [y/N] ").strip().lower() != "y":
            sys.exit(1)

    git = lambda *args: subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    files = {Path(f) for f in git("ls-files").split() if (ROOT / f).is_file()}  # your code, without caches
    for pattern in EXTRA[a]:
        files |= {p.relative_to(ROOT) for p in ROOT.glob(pattern)}
    files |= {Path(f) for f in GRADED[a] if (ROOT / f).exists()}

    home = Path(os.environ.get("COS568_HOME", ""))
    if not home.is_dir():
        sys.exit(f"Your course directory {home} does not exist. Tell the course staff.")
    dest = home / "handin" / f"a{a}"
    dest.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = dest / f"{stamp}.tar.gz"
    info = ROOT / "HANDIN_INFO"
    info.write_text(f"user: {getpass.getuser()}\ntime: {stamp}\ncommit: {git('rev-parse', 'HEAD') or 'none'}\n"
                    f"uncommitted changes: {'yes' if git('status', '--porcelain', '--untracked-files=no') else 'no'}\n")
    try:
        with tarfile.open(out, "w:gz") as tar:
            tar.add(info, arcname=info.name)
            for f in sorted(files):
                tar.add(ROOT / f, arcname=str(f))
    finally:
        info.unlink()
    latest = dest / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(out.name)

    mb = out.stat().st_size / 2**20
    print(f"Handed in {len(files)} files ({mb:.1f} MB) as {out}")
    if mb > MAX_MB:
        print(f"That is large (over {MAX_MB} MB). Record shorter profiles (--steps 1) and hand in again.")
    print("You can hand in again at any time; the last hand-in before the deadline is graded.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "1")
