# COS568 assignments. Every target loads the course environment by itself:
# there is nothing to install and nothing to source.
#
#   make check          verify your setup on a GPU node
#   make test           run the public tests and print your autograded score
#   make test C=6       ... only Coding Exercise 6, with more detail about failures
#   make test E=3       ... only the checks of Experiment 3
#   make gpu-shell      interactive shell on a GPU node (2 hours at most)
#   make handin         hand in your work
#
# A=<n> selects the assignment (default 1).
A ?= 1
PARTITION ?= interactive

SHELL := /bin/bash
ENV   := source setup/env.sh &&
# GPU work never runs on the login node: outside a Slurm job, targets go through srun.
SRUN  := srun --partition=$(PARTITION) --gres=gpu:1 --cpus-per-task=8 --mem=48G
RUN   := $(if $(SLURM_JOB_ID),,$(SRUN) --time=0:30:00)
ONLY  := $(if $(C),Coding $(C),$(if $(E),Experiment $(E)))

.PHONY: check test gpu-shell handin

check:
	@$(ENV) $(RUN) python tools/check.py

test:
	@$(ENV) $(RUN) python -m pytest -q -rs grade/test_a$(A).py $(if $(ONLY),--only="$(ONLY)" --tb=short,--tb=line)

gpu-shell:
	@$(ENV) $(SRUN) --time=2:00:00 --pty bash --rcfile setup/shellrc -i

handin:
	@$(ENV) python tools/handin.py $(A)
