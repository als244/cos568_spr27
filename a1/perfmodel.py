"""Your performance model of OLMoE on one L40 (Coding Exercises 2, 3, 8 and 9).

Everything in this file is arithmetic: no GPU and no torch. Conventions:
  * one multiply-add is 2 FLOPs;
  * sizes are in bytes and times in milliseconds;
  * bandwidth is in GB/s (1e9 bytes/s) and compute in TFLOP/s (1e12 FLOP/s).
"""
from olmoe.config import OlmoeConfig

# NVIDIA L40
PEAK_TFLOPS = 181.05     # dense bf16 matmul on the tensor cores, at the 2.49 GHz boost clock:
                         # twice the 90.5 TFLOP/s fp32 figure of the datasheet
PEAK_GBPS = 864.0        # memory bandwidth of the 48 GB of GDDR6
MEMORY_GIB = 44.4        # device memory that PyTorch can use
L2_BYTES = 96 * 2**20    # L2 cache


# --------------------------------------------------------- Coding Exercise 2
def gemm_flops(m: int, n: int, k: int) -> int:
    """FLOPs of the matrix multiply C = torch.matmul(A, B), for A: [m, k] and B: [k, n]."""
    raise NotImplementedError("Coding Exercise 2")


def gemm_bytes(m: int, n: int, k: int, itemsize: int = 2) -> int:
    """Least memory traffic of that GEMM: read A and B once, write C once."""
    raise NotImplementedError("Coding Exercise 2")


def roofline_tflops(intensity: float, peak_tflops: float = PEAK_TFLOPS,
                    peak_gbps: float = PEAK_GBPS) -> float:
    """The roofline: the highest TFLOP/s a kernel with `intensity` FLOPs per byte can reach."""
    raise NotImplementedError("Coding Exercise 2")


# --------------------------------------------------------- Coding Exercise 3
def param_count(cfg: OlmoeConfig) -> int:
    """All parameters: embedding, norms, attention, routers, experts, lm_head."""
    raise NotImplementedError("Coding Exercise 3")


def active_params(cfg: OlmoeConfig) -> int:
    """Weights that one token is multiplied by in a forward pass: the attention
    projections, the router, its top_k experts, and lm_head. (Norms are not
    matmuls, and the embedding is a lookup.)"""
    raise NotImplementedError("Coding Exercise 3")


def flops_per_token(cfg: OlmoeConfig, seqlen: int, training: bool = False) -> int:
    """Model FLOPs per token, for sequences of `seqlen` tokens.
        forward  = 2 * active_params + 2 * n_layers * seqlen * hidden_size
        training = 3 * forward        (backward costs about twice the forward)
    The second term is attention. Scores and the weighted sum cost
    4 * hidden_size FLOPs per (token, position) pair, and with a causal mask a
    token attends to seqlen / 2 positions on average.
    """
    raise NotImplementedError("Coding Exercise 3")


def rows_per_expert(cfg: OlmoeConfig, tokens: int) -> float:
    """Average number of rows in one expert's GEMM when a step processes
    `tokens` tokens, if routing is perfectly balanced."""
    raise NotImplementedError("Coding Exercise 3")


def forward_floor_ms(cfg: OlmoeConfig, tokens: int, seqlen: int) -> tuple[float, float]:
    """Two lower bounds on the time of one forward pass over `tokens` tokens
    in sequences of length `seqlen`. Returns (compute_ms, memory_ms):
        compute_ms  the forward FLOPs at peak compute;
        memory_ms   reading every weight once at peak bandwidth. Weights are
                    bf16. Count every parameter except the embedding table,
                    which is only indexed.
    The pass can be no faster than the larger of the two.
    """
    raise NotImplementedError("Coding Exercise 3")


def mfu(cfg: OlmoeConfig, seqlen: int, tokens_per_s: float, training: bool = True) -> float:
    """Model FLOPs utilization: the share of peak compute spent on the model's arithmetic."""
    return flops_per_token(cfg, seqlen, training) * tokens_per_s / (PEAK_TFLOPS * 1e12)


# -------------------------------------------------------- Coding Exercise 8
def state_bytes(cfg: OlmoeConfig, itemsize: int = 2) -> int:
    """Memory that training holds per parameter, summed over the model: the
    weight, its gradient, and AdamW's two moments, each `itemsize` bytes."""
    raise NotImplementedError("Coding Exercise 8")


def optimizer_floor_ms(cfg: OlmoeConfig, itemsize: int = 2) -> float:
    """Lower bound on the time of one AdamW step at peak bandwidth. For every
    parameter, AdamW reads the weight, the gradient and both moments, and
    writes the weight and both moments."""
    raise NotImplementedError("Coding Exercise 8")


# -------------------------------------------------------- Coding Exercise 9
def predict_step_ms(cfg: OlmoeConfig, micro_batch_size: int, seqlen: int, recompute: bool = False) -> float:
    """Predict the GPU time (ms) of one train_step on an L40, for a model with
    cfg.n_layers layers. The step's global batch is a single micro batch of
    `micro_batch_size` sequences of `seqlen` tokens each."""
    raise NotImplementedError("Coding Exercise 9")


def predict_peak_gib(cfg: OlmoeConfig, micro_batch_size: int, seqlen: int, recompute: bool = False) -> float:
    """Predict the peak GPU memory (GiB, as torch.cuda.max_memory_allocated
    reports it) of train_step for the same model and the same micro batch."""
    raise NotImplementedError("Coding Exercise 9")


# -------------------------------------------------- Challenge (optional)
def predict_step_ms_rtx5090(cfg: OlmoeConfig, micro_batch_size: int, seqlen: int, recompute: bool = False) -> float:
    """The prediction of predict_step_ms, for the other machine of the last
    Challenge: one RTX 5090, as the files in a1/rtx5090/ describe it."""
    raise NotImplementedError("Challenge: another machine")
