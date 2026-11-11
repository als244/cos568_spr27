"""Test harness: point values, GPU and checkpoint markers, and a score summary.

    @pytest.mark.points(5, "Coding 1")     point value and label, shown in the summary
    @pytest.mark.gpu                        needs a CUDA device
    @pytest.mark.weights                    needs the OLMoE checkpoint ($OLMOE_WEIGHTS)
"""
import gc
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "a1")]

_score = defaultdict(lambda: [0, 0])  # label -> [earned, possible]


def pytest_addoption(parser):
    parser.addoption("--only", help='only run the tests with this label, such as "Coding 6" or "Experiment 3"')


def pytest_configure(config):
    config.addinivalue_line("markers", "points(n, label): point value of a test")
    config.addinivalue_line("markers", "gpu: needs a CUDA device")
    config.addinivalue_line("markers", "weights: needs the OLMoE checkpoint")


def pytest_collection_modifyitems(config, items):
    only = config.getoption("--only")
    if only:
        keep = [i for i in items if (m := i.get_closest_marker("points")) and m.args[1] == only]
        config.hook.pytest_deselected(items=[i for i in items if i not in keep])
        items[:] = keep
    has_gpu = torch.cuda.is_available()
    weights = os.environ.get("OLMOE_WEIGHTS")
    has_weights = bool(weights) and Path(weights, "config.json").exists()
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(pytest.mark.skip(reason="no GPU: use `make test`, which runs on a GPU node"))
        if "weights" in item.keywords and not has_weights:
            item.add_marker(pytest.mark.skip(reason="OLMoE checkpoint not found ($OLMOE_WEIGHTS)"))


@pytest.fixture(autouse=True)
def _free_gpu_memory():
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    marker = item.get_closest_marker("points")
    if marker is None:
        return
    n, label = marker.args
    if report.when == "setup":
        _score[label][1] += n
    elif report.when == "call" and report.passed:
        _score[label][0] += n


def pytest_terminal_summary(terminalreporter):
    if not _score:
        return
    write = terminalreporter.write_line
    terminalreporter.section("score")
    earned = possible = 0
    kinds = ["Setup", "Coding", "Experiment"]

    def order(label):
        kind, _, number = label.partition(" ")
        return kinds.index(kind) if kind in kinds else len(kinds), int(number or 0)

    for label in sorted(_score, key=order):
        e, p = _score[label]
        earned, possible = earned + e, possible + p
        if p:
            write(f"  {label:<13s} {'OK  ' if e == p else 'FAIL'} {e:>3d}/{p}")
        else:
            write(f"  {label:<13s} (no points; these must pass before you start)")
    write(f"  {'Total':<13s}      {earned:>3d}/{possible}")
