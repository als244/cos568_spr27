"""Profiling helpers: GPU timing, NVTX ranges, GPU utilization, memory over time."""
import functools
import statistics
import threading
import time
from contextlib import contextmanager

import torch


@functools.cache
def _has_cuda() -> bool:
    return torch.cuda.is_available()


def in_backward() -> bool:
    """True while autograd is running a backward pass on this thread. A module
    whose forward runs then is being recomputed (see Olmoe.forward)."""
    return torch._C._current_graph_task_id() != -1


def _push(name: str):
    """Open an NVTX range on this thread."""
    if _has_cuda():
        # A range opened during a backward pass belongs to a recomputation; say so in its name.
        torch.cuda.nvtx.range_push(f"recompute.{name}" if in_backward() else name)


def _pop():
    """Close the NVTX range that this thread opened last."""
    if _has_cuda():
        torch.cuda.nvtx.range_pop()


@contextmanager
def nvtx(name: str):
    """Mark a region of the CPU timeline for Nsight Systems."""
    _push(name)
    try:
        yield
    finally:
        _pop()


def median(xs):
    """The median of a list of numbers."""
    return statistics.median(xs)


def warm_up(fn, seconds: float = 0.1, calls: int = 2):
    """Run fn() untimed until it has run `calls` times and for `seconds`.

    The first calls of anything pay for what later calls do not. Libraries are
    loaded. Memory is requested from the driver; afterwards PyTorch hands it
    out again from its own cache. The times in this course are steady-state
    times, so every script calls this before it measures.

    Two calls at least, because the first step of a new optimizer creates the
    optimizer's state: a training step finds all its memory in the cache only
    from its third call on.
    """
    start, done = time.perf_counter(), 0
    while done < calls or time.perf_counter() - start < seconds:
        fn()
        torch.cuda.synchronize()
        done += 1


def heat_up(celsius: int = 60, seconds: float = 60.0) -> float:
    """Keep the GPU busy with a large GEMM until its chip is at `celsius`
    degrees, for `seconds` at most. Returns how long that took, in seconds.

    A warm-up is not all that separates a measurement from a long run. An L40
    may draw 300 W, and a hot chip reaches that limit at a lower clock than a
    cool one: a compute-bound step takes 6 to 9% longer at 70 degrees than on
    a chip that has just started, at 40. Times that are to be compared should
    come from chips at a similar temperature, so a1/sweep.py calls this before
    every configuration. On a chip that is warm already it returns at once.
    """
    start = time.perf_counter()
    if torch.cuda.temperature() < celsius:
        a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
        while torch.cuda.temperature() < celsius and time.perf_counter() - start < seconds:
            for _ in range(10):
                torch.matmul(a, a)
            torch.cuda.synchronize()
    return time.perf_counter() - start


def cuda_time(fn, warmup: int = 3, iters: int = 10) -> list[float]:
    """Run fn() `warmup` times untimed, then `iters` times timed.

    Returns the GPU time of each timed call, in milliseconds. fn() must be
    called exactly warmup + iters times. The time must cover all the GPU work
    fn() enqueues, even if fn() returns before that work has finished, and
    cuda_time must not return until that work is done.
    """
    raise NotImplementedError("Coding Exercise 1: GPU timing")


def annotate(model: torch.nn.Module, depth: int = 3):
    """Give submodules NVTX ranges named by their path, such as "layers.3.moe".

    Annotate every submodule whose path has at most `depth` dot-separated
    components ("layers.3.moe" has 3). Do not annotate the root module.
    Open each range with _push(name) just before the submodule's forward and
    close it with _pop() just after, even if that forward raises an exception.
    Return the hook handles, so the caller can undo everything with h.remove().
    """
    raise NotImplementedError("Coding Exercise 4: NVTX annotation")


class GpuMonitor:
    """Sample the GPU's own counters while a block of code runs.

        with GpuMonitor() as mon:
            ...                  # a few seconds of steady work
        mon.gpu_util             # % of time at least one kernel was running
        mon.mem_util             # % of time device memory was being read or written
        mon.clock_mhz            # clock frequency of the GPU cores
        mon.power_w              # power draw
        mon.temp_c               # temperature of the chip, in degrees Celsius

    Each attribute is the mean of the samples. The counters come from NVML
    (the library behind nvidia-smi), which averages over a window of up to
    one second, so a block shorter than a few seconds gives meaningless numbers.
    """

    def __init__(self, interval: float = 0.1):
        self.interval, self.samples = interval, []
        self._stop = threading.Event()

    def _run(self):  # the sampling thread
        while not self._stop.wait(self.interval):
            self.samples.append((torch.cuda.utilization(), torch.cuda.memory_usage(), torch.cuda.clock_rate(),
                                 torch.cuda.power_draw() / 1e3, torch.cuda.temperature()))

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        means = [sum(column) / len(column) for column in zip(*self.samples)] if self.samples else [0.0] * 5
        self.gpu_util, self.mem_util, self.clock_mhz, self.power_w, self.temp_c = means


def memory_timeline(fn):
    """Run fn() and return how the allocated GPU memory moved while it ran.

    Returns (ms, gib): two lists with one entry per allocation or free, the
    CPU time since fn() started and the memory allocated from then on. This is
    PyTorch's own record of its allocator, the one behind memory_snapshot
    below, reduced to a single curve.
    """
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    torch.cuda.memory._record_memory_history(context=None, max_entries=1_000_000, clear_history=True)
    try:
        fn()
        torch.cuda.synchronize()
        trace = torch.cuda.memory._snapshot()["device_traces"][torch.cuda.current_device()]
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    ms, gib = [0.0], [allocated / 2**30]
    for event in trace:
        if event["action"] in ("alloc", "free_completed"):
            allocated += event["size"] if event["action"] == "alloc" else -event["size"]
            ms.append((event["time_us"] - trace[0]["time_us"]) / 1e3)
            gib.append(allocated / 2**30)
    return ms, gib


@contextmanager
def memory_snapshot(path: str, max_entries: int = 200_000):
    """Record every CUDA allocation inside the block; view at https://pytorch.org/memory_viz"""
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield
    finally:
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)
