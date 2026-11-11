"""One training step: forward, backward (with any recomputation), optimizer.

Two sizes describe a step. The *global batch size* is the number of
sequences that one optimizer update is computed from. The *micro batch size*
is the number of sequences that go through the model in one forward and
backward pass. train_step runs a global batch as a single micro batch;
accumulated_step runs it as several, and adds up their gradients.
"""
import torch
import torch.nn.functional as F

from .prof import in_backward, median, nvtx  # in_backward and median are for stage_times


def lm_loss(logits, targets, n_tokens=None):
    """The language-model loss of one micro batch.

    logits: [T, vocab], targets: [T]. The cross-entropies of the T tokens are
    added up and divided by `n_tokens`, the number of trainable tokens (tokens
    that have a target) in the step's whole global batch. Every token of a
    step therefore weighs 1 / n_tokens in the step's loss, however the global
    batch is split into micro batches or, in Assignment 2, spread over GPUs.

    By default n_tokens is T: this micro batch is the whole global batch, and
    the loss is the mean over its tokens.
    """
    n_tokens = targets.numel() if n_tokens is None else n_tokens
    return F.cross_entropy(logits.float(), targets, reduction="sum") / n_tokens


def make_optimizer(model, lr=1e-5):
    """AdamW over every parameter, with its state in the parameters' dtype.

    Assignment 1 keeps the weights and the Adam state in bf16: 8 bytes per
    parameter in total (weight, gradient, two moments).
    """
    fused = next(model.parameters()).is_cuda
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                             weight_decay=0.0, fused=fused)


def train_step(model, opt, micro_batch):
    """One optimizer step on a global batch that is a single micro batch (a Batch, see data.py).
    Returns the step's loss: the mean over its tokens."""
    with nvtx("forward"):
        loss = lm_loss(model(micro_batch.tokens, micro_batch.seqs), micro_batch.targets)
    with nvtx("backward"):
        loss.backward()
    with nvtx("optimizer"):
        opt.step()
        opt.zero_grad(set_to_none=True)
    return loss.detach()


def accumulated_step(model, opt, micro_batches):
    """One optimizer step on a global batch that is a list of micro batches.

    Each micro batch is a Batch of whole sequences. Run the forward and the
    backward pass of one micro batch at a time, so that only one micro batch's
    activations are alive at any moment, and let the optimizer step once, after
    the last one. The update must be the one that train_step would make on
    all the sequences as a single micro batch, up to rounding, whatever the
    sizes of the micro batches.

    Open the same NVTX ranges as train_step: "forward" and "backward" around
    each micro batch's passes, and "optimizer" around the update. Return the
    step's loss, detached: the mean over all the tokens of the global batch.
    """
    raise NotImplementedError("Coding Exercise 6: gradient accumulation")


def stage_names(model) -> list[str]:
    """The stages of the model, in forward order: embed, each layer, head.
    "head" is the final norm, lm_head, and the loss."""
    return ["embed"] + [f"layers.{i}" for i in range(len(model.layers))] + ["head"]


def stage_times(model, opt, micro_batch, warmup: int = 2, iters: int = 5) -> dict:
    """GPU time (ms) of every stage of train_step, median over `iters` steps.

    Returns
        {"forward":   {stage: ms for each stage in stage_names(model)},
         "backward":  {stage: ms for each stage},
         "recompute": {stage: ms for each stage},
         "optimizer": ms}
    forward    the stage's forward, in the forward pass;
    recompute  the stage's forward when it runs again during the backward pass
               (0.0 for every stage of a model that does not recompute);
    backward   the rest of the time that the backward pass spends in the stage.

    Run `warmup` untimed steps first. Do not synchronize inside a step: record
    events at the boundaries and read them after the last step. Remove any
    hooks you register before returning.
    """
    raise NotImplementedError("Coding Exercise 7: stage breakdown")


def phase_times(times: dict) -> dict[str, float]:
    """Collapse stage_times() output to totals in ms: forward, backward, recompute, optimizer."""
    return {"forward": sum(times["forward"].values()),
            "backward": sum(times["backward"].values()),
            "recompute": sum(times["recompute"].values()),
            "optimizer": times["optimizer"]}
