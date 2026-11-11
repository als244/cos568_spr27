"""OLMoE in about 200 lines of PyTorch.

    tokens -> Embedding -> [Block] x n_layers -> RMSNorm -> lm_head -> logits
    Block:   x = x + Attention(RMSNorm(x));  x = x + MoE(RMSNorm(x))

Shapes. Activations are flat: x is [T, D], with T the number of tokens in the
micro batch (all of its sequences concatenated) and D the hidden size. There
is no batch dimension and no padding; a `Seqs` (data.py) says where each
sequence is.

Numerics follow HuggingFace's OlmoeForCausalLM, so the official checkpoint
loads as-is (load.py) and the outputs match (grade/test_a1.py).
"""
from itertools import pairwise

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.varlen import varlen_attn
from torch.utils.checkpoint import checkpoint

from .config import OlmoeConfig
from .data import Seqs
from .prof import nvtx


class RMSNorm(nn.Module):
    """Root-mean-square normalization, with a learned scale per dimension."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Normalize in fp32, then scale in the input dtype (matches HF numerics).
        y = F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps)
        return self.weight * y.to(x.dtype)


def rope_tables(cfg: OlmoeConfig, seqs: Seqs, n_tokens: int, dtype):
    """Rotary position embeddings (RoPE): cos and sin of every token's rotation
    angles, as two [T, 1, head_dim] tensors. A token's position is counted
    from the start of its own sequence."""
    cu = seqs.cu_seqlens
    start = torch.repeat_interleave(cu[:-1], cu.diff(), output_size=n_tokens)  # [T] start of each token's sequence
    position = torch.arange(n_tokens, device=cu.device) - start
    d = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, device=cu.device).float() / d))
    angle = torch.outer(position.float(), inv_freq)       # [T, head_dim / 2]
    angle = torch.cat([angle, angle], dim=-1)[:, None]    # [T, 1, head_dim]: broadcasts over heads
    return angle.cos().to(dtype), angle.sin().to(dtype)


def apply_rope(x, cos, sin):
    """Rotate queries or keys by their position. x: [T, n_heads, head_dim]."""
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def causal_attention(q, k, v, seqs: Seqs):
    """Each token attends to itself and the earlier tokens of its own sequence.
    q, k, v: [T, n_heads, head_dim] -> [T, n_heads, head_dim]."""
    cu, longest = seqs.cu_seqlens, seqs.max_seqlen
    if q.is_cuda and q.dtype != torch.float32:
        # Variable-length FlashAttention over the packed tokens: no padding and
        # no attention matrix in memory. window_size=(-1, 0) means "causal".
        return varlen_attn(q, k, v, cu, cu, longest, longest, window_size=(-1, 0))
    # Reference path (CPU or fp32, used by the tests): one sequence at a time.
    out = [F.scaled_dot_product_attention(q[a:b].transpose(0, 1), k[a:b].transpose(0, 1),
                                          v[a:b].transpose(0, 1), is_causal=True).transpose(0, 1)
           for a, b in pairwise(cu.tolist())]
    return torch.cat(out)


class Attention(nn.Module):
    """Causal self-attention over the tokens of each sequence, with rotary position embeddings."""

    def __init__(self, cfg: OlmoeConfig):
        super().__init__()
        D = cfg.hidden_size
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.q_norm = RMSNorm(D, cfg.norm_eps)  # OLMoE normalizes q and k ("QK-norm")
        self.k_norm = RMSNorm(D, cfg.norm_eps)

    def forward(self, x, seqs, rope):  # x: [T, D]
        q = self.q_norm(self.q_proj(x))
        k = self.k_norm(self.k_proj(x))
        v = self.v_proj(x)
        q, k, v = (t.view(-1, self.n_heads, self.head_dim) for t in (q, k, v))
        q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        return self.o_proj(causal_attention(q, k, v, seqs).reshape(x.shape))


class MoE(nn.Module):
    """Top-k routed mixture of SwiGLU experts.

    The router sends each token to its top_k experts (of n_experts) and sums
    their outputs, weighted by the router's probabilities.

    Expert weights are stacked along dim 0, one slice per expert, so expert
    parallelism (Assignment 2) can shard them by slicing.

    impl="grouped" (default): sort the (token, expert) pairs by expert, so
        each expert's rows are contiguous, then multiply every block of rows
        by its own expert's weights with one grouped matmul per projection.
    impl="naive": a loop over the experts with a boolean mask for each, the
        way you might write it first. Same result. Assignment 1 finds out why
        it is slow.
    """

    def __init__(self, cfg: OlmoeConfig, impl: str = "grouped"):
        super().__init__()
        E, D, I = cfg.n_experts, cfg.hidden_size, cfg.intermediate_size
        self.top_k, self.impl = cfg.top_k, impl
        self.router = nn.Linear(D, E, bias=False)
        self.w_gate = nn.Parameter(torch.empty(E, I, D))
        self.w_up = nn.Parameter(torch.empty(E, I, D))
        self.w_down = nn.Parameter(torch.empty(E, D, I))

    def route(self, x):
        """x: [T, D] -> weights [T, k], expert ids [T, k]."""
        probs = self.router(x).softmax(dim=-1, dtype=torch.float)
        weights, experts = probs.topk(self.top_k, dim=-1)
        return weights.to(x.dtype), experts

    def forward(self, x):  # x: [T, D]
        with nvtx("router"):
            weights, experts = self.route(x)
        with nvtx("experts"):
            run = _experts_grouped if self.impl == "grouped" else _experts_naive
            return run(x, weights, experts, self.w_gate, self.w_up, self.w_down)


def _experts_grouped(x, weights, experts, w_gate, w_up, w_down):
    """Every token through its k experts, and the weighted sum of their outputs.
    x: [T, D]; weights, experts: [T, k]; returns [T, D]."""
    k = experts.shape[1]
    pair_expert = experts.flatten()            # expert of each of the T*k (token, slot) pairs
    order = pair_expert.argsort(stable=True)   # the pairs, grouped by expert
    token = order // k                         # token of each pair, in that order
    # ends[e]: one past the last row of expert e's block.
    ends = torch.bincount(pair_expert, minlength=w_gate.shape[0]).cumsum(0, dtype=torch.int32)

    def mm(a, w):
        """Rows ends[e-1]:ends[e] of `a`, times w[e].T, for every expert e at once.
        One kernel on Hopper-class GPUs; on the L40, PyTorch loops over the experts in C++."""
        return torch._grouped_mm(a, w.transpose(-2, -1), offs=ends)

    xs = x[token]                              # [T*k, D]: each token's row, once per expert it goes to
    y = mm(F.silu(mm(xs, w_gate)) * mm(xs, w_up), w_down) * weights.flatten()[order, None]
    return torch.zeros_like(x).index_add_(0, token, y)  # sum each token's k outputs


def _experts_naive(x, weights, experts, w_gate, w_up, w_down):
    """The same computation as _experts_grouped, one expert at a time."""
    out = torch.zeros_like(x)
    for e in range(w_gate.shape[0]):
        token, slot = torch.where(experts == e)  # the tokens routed to expert e, and in which of their k slots
        rows = x[token]
        h = F.silu(torch.matmul(rows, w_gate[e].T)) * torch.matmul(rows, w_up[e].T)
        out.index_add_(0, token, torch.matmul(h, w_down[e].T) * weights[token, slot, None])
    return out


class Block(nn.Module):
    """One layer: attention, then the MoE. Each reads a normalized copy of the
    stream x and adds its output back to x."""

    def __init__(self, cfg: OlmoeConfig, moe_impl: str = "grouped"):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.moe_norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.moe = MoE(cfg, moe_impl)

    def forward(self, x, seqs, rope):  # x: [T, D]
        x = x + self.attn(self.attn_norm(x), seqs, rope)
        return x + self.moe(self.moe_norm(x))


class Olmoe(nn.Module):
    """The whole model. With recompute=True, a training step keeps only the input
    of each layer and recomputes the layer's activations during backward."""

    def __init__(self, cfg: OlmoeConfig, dtype=torch.bfloat16, recompute: bool = False,
                 moe_impl: str = "grouped"):
        super().__init__()
        self.cfg = cfg
        self.recompute = recompute
        to = lambda m: m.to(dtype)  # cast each piece as it is built, so fp32 copies never pile up
        self.embed = to(nn.Embedding(cfg.vocab_size, cfg.hidden_size))
        self.layers = nn.ModuleList(to(Block(cfg, moe_impl)) for _ in range(cfg.n_layers))
        self.norm = to(RMSNorm(cfg.hidden_size, cfg.norm_eps))
        self.lm_head = to(nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False))

    def forward(self, tokens, seqs: Seqs):
        """tokens: [T] token ids, all sequences concatenated. seqs: where each sequence is.
        Returns logits [T, vocab]: for every token, the prediction of the token after it."""
        x = self.embed(tokens)  # [T, D]
        rope = rope_tables(self.cfg, seqs, tokens.shape[0], x.dtype)
        for layer in self.layers:
            if self.recompute and torch.is_grad_enabled():
                # Activation recomputation: keep only this layer's input, and
                # run its forward again during backward to rebuild the activations.
                raise NotImplementedError("Coding Exercise 5: activation recomputation")
            else:
                x = layer(x, seqs, rope)
        return self.lm_head(self.norm(x))

    @torch.no_grad()
    def init_weights(self, std: float = 0.02, seed: int = 0):
        """Random weights, the same every time for a given seed. Enough for timing."""
        g = torch.Generator(device=self.embed.weight.device).manual_seed(seed)
        for name, p in self.named_parameters():
            if "norm" in name:
                p.fill_(1.0)
            else:
                p.normal_(0.0, std, generator=g)


def build_model(cfg: OlmoeConfig, device="cuda", dtype=torch.bfloat16, init=True, **kw) -> Olmoe:
    """Create an Olmoe on `device`. init=False skips the random init (load.py fills the weights).
    Extra keyword arguments (recompute, moe_impl) go to Olmoe."""
    with torch.device(device):
        model = Olmoe(cfg, dtype=dtype, **kw)
    if init:
        model.init_weights()
    return model
