"""Micro batches of token sequences, stored flat.

A micro batch is the set of sequences that goes through the model in one
pass. Nothing in this codebase has a [batch, seqlen] shape: a micro batch is
all of its sequences concatenated, so tokens are [T] and activations are
[T, D], where T is the total number of tokens and D the hidden size. A `Seqs`
says where each sequence starts and ends, so the sequences of one micro batch
may have different lengths and nothing is ever padded.
"""
from itertools import accumulate
from typing import NamedTuple

import torch


class Seqs(NamedTuple):
    """Where each of the N sequences lives in a flat micro batch of T tokens."""
    cu_seqlens: torch.Tensor  # [N + 1] int32, cumulative lengths: sequence i is rows cu_seqlens[i]:cu_seqlens[i + 1]
    max_seqlen: int           # length of the longest sequence (a Python int, so nobody has to ask the GPU)


class Batch(NamedTuple):
    """One micro batch for training: T input tokens, the token that follows each, and where the sequences are."""
    tokens: torch.Tensor   # [T] int64: the input tokens of every sequence
    targets: torch.Tensor  # [T] int64: for each input token, the token that follows it
    seqs: Seqs


def make_seqs(seqlens: list[int], device="cuda") -> Seqs:
    """The Seqs of a micro batch whose sequences have these lengths, in this order."""
    cu_seqlens = torch.tensor([0, *accumulate(seqlens)], dtype=torch.int32, device=device)
    return Seqs(cu_seqlens, max(seqlens))


def pack(sequences: list[torch.Tensor], device="cuda") -> Batch:
    """Build a micro batch from 1-D token sequences of any lengths.
    A sequence of L + 1 tokens gives L inputs (all but its last token) and L targets (all but its first)."""
    tokens = torch.cat([s[:-1] for s in sequences]).to(device)
    targets = torch.cat([s[1:] for s in sequences]).to(device)
    return Batch(tokens, targets, make_seqs([len(s) - 1 for s in sequences], device))


def random_batch(seqlens: list[int], vocab: int, device="cuda", seed: int = 0) -> Batch:
    """A micro batch of random tokens: one sequence per entry of `seqlens`, so T = sum(seqlens).
    Assignment 1 needs nothing more, because speed depends on shapes, not on the text."""
    g = torch.Generator().manual_seed(seed)
    return pack([torch.randint(0, vocab, (n + 1,), generator=g) for n in seqlens], device)
