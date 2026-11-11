# Instructions for AI coding agents

This repository is used on the Princeton CS **neuronic** cluster. Before you
run any command, read and follow `/etc/agents.d/PRINCETON_CS_AGENTS.md`.

- **Never run computation on the login node.** That includes any Python that
  imports torch, the tests, and the benchmarks. Use `make test` and
  `make check` (they submit themselves to a GPU node through `srun`), or
  `sbatch slurm/job.sbatch <command>`.
- **Do not install anything.** The Python environment (Python 3.12, PyTorch
  2.13) and the model checkpoint are shared and read-only, and home quotas are
  small. `make` targets load the environment themselves; in a plain shell,
  run `source setup/env.sh`.
- Keep jobs short and release GPUs when you are done. Other students share them.
- Tensors are flat. Activations are `[T, D]` (T tokens of all sequences in the
  micro batch, concatenated; D the hidden size), never `[batch, seqlen, D]`.
  Sequence boundaries travel in a `Seqs` (see `olmoe/data.py`).
- Say "global batch size" (sequences per optimizer step) or "micro batch
  size" (sequences per forward and backward pass), never just "batch size".
- Do not edit `grade/` to make tests pass. Grading uses the staff's copy of
  the tests, plus hidden ones.
