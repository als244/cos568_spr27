# COS568: Systems and Machine Learning (Spring 2027) — Assignments

One codebase for the three programming assignments of the course. They are
built around the language model
[OLMoE-1B-7B](https://huggingface.co/allenai/OLMoE-1B-7B-0924) and run on the
neuronic cluster, on NVIDIA L40 GPUs.

| # | Topic | GPUs | Handout |
|---|-------|------|---------|
| 1 | Performance profiling | 1 | [assignments/a1.md](assignments/a1.md) |
| 2 | Distributed fine-tuning: data and expert parallelism | 1–8 | _coming_ |
| 3 | Inference with vLLM and SGLang | 1–8 | _coming_ |

## Start here

Read the handout of the current assignment. (It is also on the course
website, where it is nicer to read.) It is a walkthrough: it tells you what
to do, in order. Everything runs through a handful of commands:

```
make check        # is everything set up? (runs on a GPU node)
make test         # run the public tests and print your score
make test C=6     # ... only Coding Exercise 6, with more detail about failures
make test E=3     # ... only the checks of Experiment 3
make gpu-shell    # a shell on a GPU node, for experiments (2 hours at most)
make handin       # hand in your work
git pull --no-rebase --no-edit   # get corrections and the next assignment (commit your work first)
```

There is nothing to install and nothing to source: every `make` target loads
the course environment itself. Never run GPU work on the login node.
`make check`, `make test` and `make gpu-shell` take care of that by
submitting themselves to a GPU node, and for anything long there is a batch
job: `sbatch slurm/job.sbatch python a1/sweep.py`.

## What you do, and where

Each assignment asks for three kinds of work. The handout numbers them
separately.

| Kind | Where | How it is graded |
|------|-------|------------------|
| **Coding Exercises** | the functions that raise `NotImplementedError`. In Assignment 1 they are in `olmoe/prof.py`, `olmoe/train.py`, `olmoe/model.py` and `a1/perfmodel.py` | by `make test`, and by hidden tests of the same kind after you hand in |
| **Experiments** | you run the scripts in `a1/`, which write to `results/a1/` and `reports/a1/` | `make test` checks the files |
| **Written Questions** | `a1/answers.md` | by a person |

Only those files are graded. Do not change `grade/`: we grade with our own
copy of the tests.

## Layout

```
olmoe/              the model and its training code, shared by all assignments
  config.py         the shape of the model
  data.py           micro batches: flat token tensors, and where the sequences are
  model.py          OLMoE: attention, router, experts
  train.py          the training step
  prof.py           timing, NVTX ranges, GPU counters
  load.py           load and save HuggingFace-format checkpoints
a1/                 Assignment 1
  perfmodel.py      your performance model
  answers.md        your written answers
  *.py, nsys.sh     the experiments; each file starts with how to run it
assignments/        the handouts
grade/              the public tests, and the staff measurements they use
setup/env.sh        the course environment; `make` loads it for you
slurm/job.sbatch    runs any command as a batch job
tools/              what `make check` and `make handin` run
results/, reports/  written by your experiments; not in git
```

Two conventions hold everywhere.

**Tensors are flat.** The sequences that go through the model in one pass
are concatenated, so tokens are `[T]` and activations are `[T, D]`. There is
no `[batch, seqlen]` shape and no padding. `olmoe/data.py` explains it in a
few lines.

**A batch size is always one of two.** The *global batch size* is the number
of sequences that one optimizer step is computed from. The *micro batch
size* is the number of sequences in one forward and backward pass. Code,
options and file columns say which one they mean.

## The environment

Everyone uses the same prebuilt, read-only environment: Python 3.12 and
PyTorch 2.13, with the exact version of every package recorded in `uv.lock`.
`setup/env.sh` only points your shell at it. To use it outside `make`, run
`source setup/env.sh`. To build the same environment on another Linux machine
with an NVIDIA GPU, install [uv](https://docs.astral.sh/uv/) and run `uv sync`.

If you let an AI coding agent run commands on the cluster, it must follow
[`AGENTS.md`](AGENTS.md).
