"""
Shared timing utilities for kernel benchmarks.

Extracts the common ``time_it_cuda`` / ``time_it_cpu`` pattern that was
duplicated across every kernel benchmark script.
"""

from __future__ import annotations

import time
from typing import Callable

import torch


def time_it_cuda(fn: Callable, iters: int = 50, warmup: int = 10) -> float:
    """
    Time a GPU function using CUDA events.

    Returns average milliseconds per call.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


def time_it_cpu(fn: Callable, iters: int = 50, warmup: int = 10) -> float:
    """
    Time a CPU function using ``time.perf_counter``.

    Returns average milliseconds per call.
    """
    for _ in range(warmup):
        fn()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    total = time.perf_counter() - t0

    return (total * 1000) / iters
