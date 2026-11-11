"""Public tests for Assignment 1. Run them with `make test`.

The hidden tests check the same properties with other shapes, seeds and
configurations, so code that passes these honestly will pass those too.
"""
import copy
import csv
import statistics
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import perfmodel as pm
from olmoe import OLMOE_1B_7B, TINY, build_model, load_model, make_seqs, pack, prof, random_batch, save_hf
from olmoe.model import Block, causal_attention
from olmoe.train import accumulated_step, lm_loss, make_optimizer, stage_names, stage_times, train_step

points = pytest.mark.points
gpu = pytest.mark.gpu
weights = pytest.mark.weights
ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------- helpers
def _gpu_ms(fn):
    """GPU time of fn(), measured directly with CUDA events."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def _busy_cycles(ms: float) -> int:
    """Argument that makes torch.cuda._sleep keep the GPU busy for about `ms` milliseconds."""
    n = 5_000_000
    _gpu_ms(lambda: torch.cuda._sleep(n))  # warm up
    return int(ms * n / _gpu_ms(lambda: torch.cuda._sleep(n)))


def _grads(model, batch):
    model.zero_grad()
    lm_loss(model(batch.tokens, batch.seqs), batch.targets).backward()
    return {n: p.grad.clone() for n, p in model.named_parameters()}


def _hf_logits(hf, batch):
    """HuggingFace's model, run on one sequence at a time. Returns [T, vocab]."""
    cu = batch.seqs.cu_seqlens.tolist()
    return torch.cat([hf(batch.tokens[a:b][None]).logits[0] for a, b in zip(cu, cu[1:])])


def _cpu_model(cfg=TINY, **kw):
    torch.manual_seed(0)
    return build_model(cfg, device="cpu", dtype=torch.float32, **kw)


# ------------------------------------------------- Setup (provided; 0 points)
def _tiny_hf(tmp_path):
    """A small random HuggingFace Olmoe, saved to tmp_path/hf."""
    transformers = pytest.importorskip("transformers")
    hf_cfg = transformers.OlmoeConfig(
        vocab_size=TINY.vocab_size, hidden_size=TINY.hidden_size,
        intermediate_size=TINY.intermediate_size, num_hidden_layers=TINY.n_layers,
        num_attention_heads=TINY.n_heads, num_key_value_heads=TINY.n_heads,
        num_experts=TINY.n_experts, num_experts_per_tok=TINY.top_k, tie_word_embeddings=False)
    torch.manual_seed(0)
    hf = transformers.OlmoeForCausalLM(hf_cfg).eval()
    hf.save_pretrained(tmp_path / "hf", safe_serialization=True)
    return hf


@points(0, "Setup")
@pytest.mark.parametrize("impl", ["grouped", "naive"])
def test_matches_huggingface_tiny(tmp_path, impl):
    hf = _tiny_hf(tmp_path)
    ours = load_model(tmp_path / "hf", device="cpu", dtype=torch.float32, moe_impl=impl).eval()
    batch = random_batch([16, 5, 9], TINY.vocab_size, device="cpu")  # three sequences, three lengths
    with torch.no_grad():
        torch.testing.assert_close(ours(batch.tokens, batch.seqs), _hf_logits(hf, batch), atol=1e-4, rtol=1e-4)


@points(0, "Setup")
def test_moe_implementations_agree():
    a = _cpu_model()
    b = copy.deepcopy(a)
    for layer in b.layers:
        layer.moe.impl = "naive"
    batch = random_batch([32, 20], TINY.vocab_size, device="cpu")
    ga, gb = _grads(a, batch), _grads(b, batch)
    for name in ga:
        torch.testing.assert_close(ga[name], gb[name], atol=1e-6, rtol=1e-4, msg=name)


@points(0, "Setup")
def test_save_hf_roundtrip(tmp_path):
    transformers = pytest.importorskip("transformers")
    _tiny_hf(tmp_path)
    ours = load_model(tmp_path / "hf", device="cpu", dtype=torch.float32).eval()
    save_hf(ours, tmp_path / "out", src_dir=tmp_path / "hf")
    back = transformers.OlmoeForCausalLM.from_pretrained(tmp_path / "out").eval()
    batch = random_batch([16], TINY.vocab_size, device="cpu")
    with torch.no_grad():
        torch.testing.assert_close(ours(batch.tokens, batch.seqs), _hf_logits(back, batch), atol=1e-4, rtol=1e-4)


@points(0, "Setup")
def test_in_backward():
    seen = []
    y = torch.ones(2, requires_grad=True) * 2
    y.register_hook(lambda grad: seen.append(prof.in_backward()))
    assert not prof.in_backward()
    y.sum().backward()
    assert seen == [True]


@points(0, "Setup")
@gpu
def test_varlen_attention_matches_reference():
    """FlashAttention over packed sequences of different lengths, against one sequence at a time."""
    seqs = make_seqs([7, 64, 1, 33])
    q, k, v = (torch.randn(105, 4, 32, device="cuda") for _ in range(3))
    fast = causal_attention(q.bfloat16(), k.bfloat16(), v.bfloat16(), seqs)
    slow = causal_attention(q, k, v, seqs)
    torch.testing.assert_close(fast.float(), slow, atol=3e-2, rtol=3e-2)


@points(0, "Setup")
@gpu
@weights
def test_checkpoint_matches_huggingface():
    """The first 4 layers of the real checkpoint, in fp32, against HuggingFace."""
    transformers = pytest.importorskip("transformers")
    from olmoe.load import default_weights_path
    batch = random_batch([64, 24], OLMOE_1B_7B.vocab_size)
    with torch.no_grad():
        ours = load_model(n_layers=4, dtype=torch.float32).eval()(batch.tokens, batch.seqs)
        hf = transformers.OlmoeForCausalLM.from_pretrained(
            default_weights_path(), num_hidden_layers=4, torch_dtype=torch.float32).cuda().eval()
        torch.testing.assert_close(ours, _hf_logits(hf, batch), atol=5e-3, rtol=5e-3)


@points(0, "Setup")
@gpu
@weights
def test_checkpoint_models_english():
    """The full model, in bf16, should find easy English text unsurprising."""
    from tokenizers import Tokenizer
    from olmoe.load import default_weights_path
    text = ("The capital of France is Paris. The capital of Germany is Berlin. "
            "The capital of Italy is Rome. The capital of Spain is Madrid. "
            "The capital of Japan is Tokyo. The capital of Egypt is Cairo.")
    tok = Tokenizer.from_file(str(default_weights_path() / "tokenizer.json"))
    batch = pack([torch.tensor(tok.encode(text).ids)])
    with torch.no_grad():
        loss = lm_loss(load_model().eval()(batch.tokens, batch.seqs), batch.targets).item()
    assert loss < 3.0, f"{loss:.2f} nats per token on easy English (random weights give about 10.8)"


# -------------------------------------------------------- Coding Exercise 1: timing
@points(9, "Coding 1")
@gpu
def test_cuda_time_measures_gpu_time():
    cycles, calls = _busy_cycles(20), []

    def fn():  # enqueues about 20 ms of GPU work and returns immediately
        calls.append(1)
        torch.cuda._sleep(cycles)

    times = prof.cuda_time(fn, warmup=2, iters=5)
    assert len(calls) == 7, "fn must run exactly warmup + iters times"
    assert len(times) == 5, "return one time per timed call"
    for t in times:
        assert t == pytest.approx(20, rel=0.2), f"measured {t:.2f} ms for 20 ms of GPU work"


@points(4, "Coding 1")
@gpu
def test_cuda_time_waits_for_the_gpu():
    cycles = _busy_cycles(50)
    torch.cuda.synchronize()
    prof.cuda_time(lambda: torch.cuda._sleep(cycles), warmup=0, iters=1)
    assert torch.cuda.current_stream().query(), "the GPU was still busy when cuda_time returned"


# -------------------------------------------------- Coding Exercise 2: one operation
@points(4, "Coding 2")
def test_gemm_model():
    assert pm.gemm_flops(128, 256, 512) == 2 * 128 * 256 * 512
    assert pm.gemm_bytes(128, 256, 512) == 2 * (128 * 512 + 512 * 256 + 128 * 256)
    assert pm.gemm_bytes(8, 8, 8, itemsize=4) == 4 * 3 * 64
    assert pm.roofline_tflops(1.0, peak_tflops=100.0, peak_gbps=1000.0) == pytest.approx(1.0)
    assert pm.roofline_tflops(1e6, peak_tflops=100.0, peak_gbps=1000.0) == pytest.approx(100.0)
    ridge = pm.PEAK_TFLOPS * 1e3 / pm.PEAK_GBPS
    assert pm.roofline_tflops(ridge) == pytest.approx(pm.PEAK_TFLOPS)
    assert pm.roofline_tflops(ridge / 2) == pytest.approx(pm.PEAK_TFLOPS / 2)


# ---------------------------------------------- Coding Exercise 3: model accounting
@points(3, "Coding 3")
def test_param_count():
    for cfg in [TINY, TINY.with_(n_layers=3, n_experts=4, top_k=1)]:
        model = _cpu_model(cfg, init=False)
        assert pm.param_count(cfg) == sum(p.numel() for p in model.parameters())
    assert pm.param_count(OLMOE_1B_7B) == 6_919_161_856


@points(4, "Coding 3")
def test_flops_per_token():
    c = OLMOE_1B_7B
    active = 16 * (4 * 2048**2 + 64 * 2048 + 8 * 3 * 2048 * 1024) + 2048 * 50304
    assert pm.active_params(c) == active
    forward = 2 * active + 2 * 16 * 1024 * 2048
    assert pm.flops_per_token(c, 1024) == forward
    assert pm.flops_per_token(c, 1024, training=True) == 3 * forward
    assert pm.rows_per_expert(c, 4 * 1024) == pytest.approx(512.0)
    assert pm.rows_per_expert(TINY, 100) == pytest.approx(25.0)


@points(4, "Coding 3")
def test_forward_floor():
    c = OLMOE_1B_7B
    compute, memory = pm.forward_floor_ms(c, tokens=512, seqlen=512)
    assert compute == pytest.approx(512 * pm.flops_per_token(c, 512) / (pm.PEAK_TFLOPS * 1e12) * 1e3)
    assert memory == pytest.approx(2 * (6_919_161_856 - 50304 * 2048) / (pm.PEAK_GBPS * 1e9) * 1e3)
    compute4, memory4 = pm.forward_floor_ms(c.with_(n_layers=4), tokens=8192, seqlen=2048)
    assert compute4 > memory4 > 0


# ----------------------------------------------------- Coding Exercise 4: NVTX ranges
@pytest.fixture
def nvtx_log(monkeypatch):
    log = []
    monkeypatch.setattr(prof, "_push", lambda name: log.append(name))
    monkeypatch.setattr(prof, "_pop", lambda: log.append(None))
    return log


def _check_nesting(log):
    depth = 0
    for name in log:
        depth += 1 if name is not None else -1
        assert depth >= 0, "a range was closed before it was opened"
    assert depth == 0, "some ranges were never closed"


@points(6, "Coding 4")
def test_annotate(nvtx_log):
    model = _cpu_model()
    batch = random_batch([8], TINY.vocab_size, device="cpu")
    handles = prof.annotate(model, depth=3)
    model(batch.tokens, batch.seqs)
    names = [n for n in nvtx_log if n is not None]
    for want in ["embed", "layers.0", "layers.0.attn", "layers.1.moe", "layers.1.moe_norm", "norm", "lm_head"]:
        assert names.count(want) == 1, f"expected exactly one range named {want!r}"
    assert "layers.0.attn.q_proj" not in names, "depth=3 must not annotate 4-component paths"
    assert names.index("layers.0") < names.index("layers.0.attn") < names.index("router")
    _check_nesting(nvtx_log)

    nvtx_log.clear()
    shallow = _cpu_model()
    prof.annotate(shallow, depth=1)
    shallow(batch.tokens, batch.seqs)
    assert {n for n in nvtx_log if n} == {"embed", "norm", "lm_head", "router", "experts"}

    for h in handles:
        h.remove()
    nvtx_log.clear()
    model(batch.tokens, batch.seqs)
    assert {n for n in nvtx_log if n} == {"router", "experts"}, "h.remove() must undo annotate"


@points(2, "Coding 4")
def test_annotate_closes_ranges_on_exceptions(nvtx_log):
    class Boom(nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")

    model = nn.Sequential(nn.Identity(), nn.Sequential(nn.Identity(), Boom()))
    prof.annotate(model, depth=2)
    with pytest.raises(RuntimeError, match="boom"):
        model(torch.zeros(1))
    assert [n for n in nvtx_log if n] == ["0", "1", "1.0", "1.1"]
    _check_nesting(nvtx_log)


# ------------------------------------------------ Coding Exercise 5: recomputation
@points(5, "Coding 5")
def test_recompute_gives_the_same_gradients():
    model = _cpu_model()
    batch = random_batch([32, 12], TINY.vocab_size, device="cpu")
    expected = _grads(model, batch)
    model.recompute = True
    got = _grads(model, batch)
    for name in expected:
        torch.testing.assert_close(got[name], expected[name], atol=1e-6, rtol=1e-4, msg=name)


@points(3, "Coding 5")
def test_recompute_saves_fewer_tensors():
    model = _cpu_model(TINY.with_(n_layers=4))
    batch = random_batch([64, 64], TINY.vocab_size, device="cpu")

    def saved_bytes():
        total = [0]

        def pack(t):
            total[0] += t.numel() * t.element_size()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            loss = lm_loss(model(batch.tokens, batch.seqs), batch.targets)
        loss.backward()
        return total[0]

    full = saved_bytes()
    model.recompute = True
    assert saved_bytes() < 0.5 * full, "with recompute, far fewer activations should be saved"


@points(4, "Coding 5")
@gpu
def test_recompute_lowers_activation_memory():
    cfg = OLMOE_1B_7B.with_(n_layers=2, vocab_size=8192)
    batch = random_batch([2048, 2048], cfg.vocab_size)
    held = []
    for recompute in (False, True):
        model = build_model(cfg, recompute=recompute)
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        loss = lm_loss(model(batch.tokens, batch.seqs), batch.targets)
        held.append(torch.cuda.memory_allocated() - before)  # activations kept for backward
        loss.backward()
        del model, loss
        torch.cuda.empty_cache()
    assert held[1] < 0.5 * held[0], f"activations held: {held[0] / 2**20:.0f} MiB -> {held[1] / 2**20:.0f} MiB"


# ------------------------------------------- Coding Exercise 6: gradient accumulation
class _CountingSGD(torch.optim.SGD):
    """Plain SGD that counts its steps. With lr = 1, a step moves every weight by exactly its gradient."""
    steps = 0

    def step(self, *args, **kwargs):
        self.steps += 1
        return super().step(*args, **kwargs)


def _update(step, model, global_batch):
    """What `step(model, optimizer, global_batch)` does to a copy of the model: the change of every weight, and the loss."""
    model = copy.deepcopy(model)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    opt = _CountingSGD(model.parameters(), lr=1.0)
    loss = step(model, opt, global_batch)
    assert opt.steps == 1, f"the optimizer must step once per global batch; it stepped {opt.steps} times"
    assert all(p.grad is None or not p.grad.any() for p in model.parameters()), "clear the gradients after the step"
    return {name: p.detach() - before[name] for name, p in model.named_parameters()}, float(loss)


def _same_update(model, sequences, split):
    """Compare accumulated_step on `sequences`, in micro batches of `split` sequences, with train_step on all of them."""
    one = pack(sequences, device="cpu")
    starts = [sum(split[:i]) for i in range(len(split))]
    several = [pack(sequences[a:a + n], device="cpu") for a, n in zip(starts, split)]
    expected, loss = _update(train_step, model, one)
    got, accumulated_loss = _update(accumulated_step, model, several)
    assert accumulated_loss == pytest.approx(loss, rel=1e-4), "return the mean loss over all tokens of the global batch"
    for name in expected:
        # The same terms, added up in a different order: equal up to rounding, not bit for bit.
        torch.testing.assert_close(got[name], expected[name], atol=1e-6, rtol=1e-4, msg=name)


@pytest.mark.parametrize("lengths, split", [
    pytest.param([16] * 6, [2, 2, 2], marks=points(2, "Coding 6"), id="micro batches of the same size"),
    pytest.param([12, 30, 7, 21, 16, 9], [1, 4, 1], marks=points(3, "Coding 6"), id="micro batches of different sizes")])
def test_accumulated_step_makes_the_same_update(lengths, split):
    g = torch.Generator().manual_seed(1)
    sequences = [torch.randint(0, TINY.vocab_size, (n + 1,), generator=g) for n in lengths]
    _same_update(_cpu_model(), sequences, split)


@points(3, "Coding 6")
@gpu
def test_accumulated_step_lowers_peak_memory():
    cfg = OLMOE_1B_7B.with_(n_layers=2, vocab_size=8192)
    model = build_model(cfg)
    opt = make_optimizer(model)
    one = [random_batch([2048] * 8, cfg.vocab_size)]  # a global batch of 8 sequences, as one micro batch
    four = [random_batch([2048] * 2, cfg.vocab_size, seed=i) for i in range(4)]  # ... and as four
    accumulated_step(model, opt, four)  # the first step allocates the optimizer's state
    peaks = []
    for micro_batches in (one, four):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        accumulated_step(model, opt, micro_batches)
        peaks.append((torch.cuda.max_memory_allocated() - before) / 2**30)
    assert peaks[1] < 0.7 * peaks[0], \
        f"memory above the training state: {peaks[0]:.1f} GiB as one micro batch, {peaks[1]:.1f} GiB as four"


# ------------------------------------------------- Coding Exercise 7: stage breakdown
class _Busy(torch.autograd.Function):
    """Identity that keeps the GPU busy for `fwd` cycles going forward and `bwd` going backward."""

    @staticmethod
    def forward(ctx, x, fwd, bwd):
        ctx.bwd = bwd
        torch.cuda._sleep(fwd)
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        torch.cuda._sleep(ctx.bwd)
        return grad, None, None


class _SlowBlock(Block):
    def forward(self, x, seqs, rope):
        return super().forward(_Busy.apply(x, self.fwd, self.bwd), seqs, rope)


@points(6, "Coding 7")
@gpu
def test_stage_times_cover_the_step():
    cfg = OLMOE_1B_7B.with_(n_layers=2, vocab_size=8192)
    model = build_model(cfg)
    opt = make_optimizer(model)
    batch = random_batch([512] * 4, cfg.vocab_size)
    t = stage_times(model, opt, batch, warmup=2, iters=3)
    names = stage_names(model)
    assert set(t) == {"forward", "backward", "recompute", "optimizer"}
    for phase in ("forward", "backward", "recompute"):
        assert set(t[phase]) == set(names), f"{phase}: one entry per stage"
    assert all(v > 0 for v in [*t["forward"].values(), *t["backward"].values(), t["optimizer"]])
    assert all(v == 0 for v in t["recompute"].values()), "this model does not recompute"
    assert not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()), \
        "remove your hooks before returning"
    step = statistics.median(_gpu_ms(lambda: train_step(model, opt, batch)) for _ in range(3))
    total = sum(t["forward"].values()) + sum(t["backward"].values()) + t["optimizer"]
    assert total == pytest.approx(step, rel=0.25), "the stages should add up to the whole step"


@points(7, "Coding 7")
@gpu
@pytest.mark.parametrize("recompute", [False, True], ids=["plain", "recompute"])
def test_stage_times_attribute_time_to_the_right_stage(recompute):
    """Make layers.1 cost an extra 30 ms forward and 50 ms backward, and look for it."""
    model = build_model(TINY, recompute=recompute)
    slow = model.layers[1]
    slow.__class__ = _SlowBlock
    slow.fwd, slow.bwd = _busy_cycles(30), _busy_cycles(50)
    opt = make_optimizer(model)
    t = stage_times(model, opt, random_batch([64, 40], TINY.vocab_size), warmup=1, iters=3)
    extra = {phase: t[phase]["layers.1"] - t[phase]["layers.0"] for phase in ("forward", "backward", "recompute")}
    assert extra["forward"] == pytest.approx(30, rel=0.2), "forward time of the slow layer"
    assert extra["backward"] == pytest.approx(50, rel=0.2), "backward time of the slow layer"
    if recompute:  # the slow layer's forward runs again during backward: that is recompute time
        assert extra["recompute"] == pytest.approx(30, rel=0.2), "recompute time of the slow layer"
        assert t["recompute"]["embed"] == 0 and t["recompute"]["head"] == 0
    else:
        assert all(v == 0 for v in t["recompute"].values())


# ------------------------------------------ Coding Exercise 8: memory-bound bounds
@points(4, "Coding 8")
def test_training_state_and_optimizer_floor():
    c = OLMOE_1B_7B.with_(n_layers=4)
    params = pm.param_count(c)
    assert pm.state_bytes(c) == 8 * params
    assert pm.state_bytes(c, itemsize=4) == 16 * params
    assert pm.optimizer_floor_ms(c) == pytest.approx(14 * params / (pm.PEAK_GBPS * 1e9) * 1e3)
    assert pm.optimizer_floor_ms(c, itemsize=4) == pytest.approx(2 * pm.optimizer_floor_ms(c))


# ------------------------------------------------- Coding Exercise 9: the predictor
TRUTH = ROOT / "grade" / "a1_truth.csv"  # measured by the course staff on an L40


def _hit_rate(predict, column, tolerance):
    """Share of the measured configurations that `predict` gets within `tolerance`."""
    hits = []
    with open(TRUTH) as f:
        for row in csv.DictReader(f):
            cfg = OLMOE_1B_7B.with_(n_layers=int(row["n_layers"]))
            guess = predict(cfg, int(row["micro_batch_size"]), int(row["seqlen"]), row["recompute"] == "1")
            hits.append(abs(guess - float(row[column])) <= tolerance * float(row[column]))
    return sum(hits) / len(hits)


@points(4, "Coding 9")
def test_predict_step_ms_roughly():
    rate = _hit_rate(pm.predict_step_ms, "step_ms", 0.30)
    assert rate >= 0.8, f"only {rate:.0%} of the step times are within 30%"


@points(6, "Coding 9")
def test_predict_step_ms_closely():
    rate = _hit_rate(pm.predict_step_ms, "step_ms", 0.15)
    assert rate >= 0.8, f"only {rate:.0%} of the step times are within 15%"


@points(4, "Coding 9")
def test_predict_peak_gib_roughly():
    rate = _hit_rate(pm.predict_peak_gib, "peak_gib", 0.20)
    assert rate >= 0.8, f"only {rate:.0%} of the peak memories are within 20%"


@points(6, "Coding 9")
def test_predict_peak_gib_closely():
    rate = _hit_rate(pm.predict_peak_gib, "peak_gib", 0.08)
    assert rate >= 0.8, f"only {rate:.0%} of the peak memories are within 8%"


# ------------------------------------------------------------- Experiments 2 to 7
# These read the files your experiments wrote to results/a1/ and check that
# they are complete and plausible.
RESULTS = ROOT / "results" / "a1"


def _rows(name):
    path = RESULTS / name
    assert path.exists(), f"results/a1/{name} is missing: run the experiment that writes it"
    with open(path) as f:
        return list(csv.DictReader(f))


def _ranges(name):
    return {row["range"]: row for row in _rows(f"{name}_ranges.csv")}


@points(5, "Experiment 2")
def test_results_one_operation():
    gemm = _rows("gemm.csv")
    assert len({row["shape"] for row in gemm}) == 5 and len(gemm) == 80, "gemm.csv is incomplete"
    best = max(float(row["tflops"]) for row in gemm)
    assert 60 < best < 1.05 * pm.PEAK_TFLOPS, f"the fastest GEMM ran at {best:.0f} TFLOP/s"
    assert len(_rows("gemm_sustained.csv")) >= 5 and len(_rows("ops.csv")) >= 30
    regimes = {row["workload"].split(",")[0]: row for row in _rows("regimes.csv")}
    assert float(regimes["add two 1 GiB tensors"]["mem_util"]) > 60
    assert float(regimes["large GEMM"]["mem_util"]) < 50
    assert float(regimes["small GEMM"]["gpu_util"]) < 85


@points(5, "Experiment 3")
def test_results_forward_profiles():
    small, large, naive = _ranges("fwd_small"), _ranges("fwd_large"), _ranges("fwd_small_naive")
    for ranges in (small, large, naive):
        assert {"forward", "experts", "router", "layers.*.attn", "lm_head", "embed"} <= set(ranges)
        assert float(ranges["layers.*"]["instances_per_step"]) == 16, "profile all 16 layers"
    ms = lambda ranges: float(ranges["forward"]["gpu_ms_per_step"])
    ops = lambda ranges: float(ranges["forward"]["gpu_ops_per_step"])
    assert ms(large) > 3 * ms(small), "fwd_large should have 32 times the tokens of fwd_small"
    assert ms(naive) > 1.5 * ms(small) and ops(naive) > 2 * ops(small), "is fwd_small_naive really naive?"


@points(5, "Experiment 4")
def test_results_micro_batches():
    rows = _rows("micro_batches.csv")
    ran = [row for row in rows if row["step_ms"]]
    assert len({row["micro_batch_size"] for row in rows}) >= 4, "try at least four micro batch sizes"
    assert {row["recompute"] for row in ran} == {"0", "1"}, "measure with and without recomputation"
    for recompute in "01":
        peak = {int(row["micro_batch_size"]): float(row["peak_gib"]) for row in ran if row["recompute"] == recompute}
        assert peak[min(peak)] < peak[max(peak)], "a smaller micro batch should need less memory"
    tokens_per_s = [float(row["tokens_per_s"]) for row in ran]
    assert 5_000 < min(tokens_per_s) and max(tokens_per_s) < 200_000, "implausible throughput"
    assert (RESULTS / "micro_batches.png").exists(), "results/a1/micro_batches.png is missing"


@points(5, "Experiment 5")
def test_results_stages():
    plain = {row["stage"]: row for row in _rows("stages.csv")}
    recompute = {row["stage"]: row for row in _rows("stages_rc.csv")}
    layers = [stage for stage in plain if stage.startswith("layers.")]
    assert len(layers) == 4 and set(plain) == set(recompute), "use --layers 4 for both runs"
    for stage in layers:
        fwd, bwd = float(plain[stage]["forward_ms"]), float(plain[stage]["backward_ms"])
        assert 1.2 < bwd / fwd < 3.5, f"{stage}: backward / forward = {bwd / fwd:.2f}"
        assert float(plain[stage]["recompute_ms"]) == 0, "stages.csv should be the run without --recompute"
        again = float(recompute[stage]["recompute_ms"])
        assert again == pytest.approx(fwd, rel=0.5), f"{stage}: recomputation should cost about one forward"
        assert float(recompute[stage]["backward_ms"]) == pytest.approx(bwd, rel=0.3), \
            f"{stage}: recompute time should not be counted as backward time"


@points(5, "Experiment 6")
def test_results_training_profiles():
    train, recompute, naive = _ranges("train"), _ranges("train_rc"), _ranges("train_naive")
    per_step = lambda ranges, kind: float(ranges[kind]["instances_per_step"])
    assert per_step(train, "layers.*") == 4, "use --layers 4"
    assert "recompute.layers.*" not in train, "train should be the recording without --recompute"
    assert per_step(recompute, "recompute.layers.*") == 4, "train_rc should recompute every layer once per step"
    assert float(naive["step"]["gpu_ms_per_step"]) > 2 * float(train["step"]["gpu_ms_per_step"])
    assert float(train["(backward)"]["gpu_ms_per_step"]) > float(train["forward"]["gpu_ms_per_step"])
    for name in ("train", "train_rc"):
        assert (ROOT / "reports" / "a1" / f"{name}.pickle").exists(), f"reports/a1/{name}.pickle is missing"


@points(5, "Experiment 7")
def test_results_sweep():
    rows = _rows("sweep.csv")
    key = lambda row: (row["n_layers"], row["micro_batch_size"], row["seqlen"], row["recompute"])
    assert len({key(row) for row in rows}) >= 30, "measure at least 30 configurations"
    assert len({row["n_layers"] for row in rows}) >= 2, "vary the number of layers"
    assert {row["recompute"] for row in rows} == {"0", "1"}, "measure with and without recomputation"
    # Where you measured a configuration that the staff also measured, the numbers should agree.
    with open(TRUTH) as f:
        truth = {key(row): row for row in csv.DictReader(f)}
    shared = [(row, truth[key(row)]) for row in rows if key(row) in truth]
    for column, tolerance in (("step_ms", 0.30), ("peak_gib", 0.15)):
        off = [key(mine) for mine, ref in shared
               if abs(float(mine[column]) - float(ref[column])) > tolerance * float(ref[column])]
        assert len(off) <= 0.2 * len(shared), f"{column} is far from the staff's measurement for {off}"
