# COS568 environment. `make` loads this for you; to use the environment in
# your own shell, run:   source setup/env.sh
#
# It only sets variables that point at a shared, read-only Python environment
# (Python 3.12, PyTorch 2.13) and at the course data. Nothing is installed in
# your home directory.

# ---- course staff: set this once, before release -----------------------------
export COS568_ROOT="${COS568_ROOT:-/n/fs/cos568}"
# ------------------------------------------------------------------------------
export OLMOE_WEIGHTS="${OLMOE_WEIGHTS:-$COS568_ROOT/shared/weights/OLMoE-1B-7B-0924}"
export COS568_HOME="${COS568_HOME:-$COS568_ROOT/users/$USER}"  # your course directory (hand-ins, 50 GB)

_repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_bin="$COS568_ROOT/shared/envs/cos568/bin"
case ":$PATH:" in *":$_bin:"*) ;; *) export PATH="$_bin:$PATH" ;; esac
export PYTHONPATH="$_repo"
# Keep caches out of your home quota. Everything the course needs is already on disk.
export HF_HUB_OFFLINE=1
export HF_HOME="/tmp/$USER/hf"
export MPLCONFIGDIR="/tmp/$USER/matplotlib"
export PYTHONPYCACHEPREFIX="/tmp/$USER/pycache"
unset _repo _bin
