#!/bin/bash
#SBATCH --job-name=cos568
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j.out
#
# Run any command on one GPU in the course environment, without keeping a
# terminal open. From the repo root:
#
#     sbatch slurm/job.sbatch python a1/sweep.py --layers 4
#
# Watch it with `squeue -u $USER`. The output goes to slurm-<jobid>.out.
cd "$SLURM_SUBMIT_DIR"
source setup/env.sh
exec "$@"
