# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# See licenses/CUTLASS-BSD-3-Clause.txt. No new kernel math in this package.
from threading import Lock
from tileirblas_blocked import validate_gemm, gemm_nt as _blocked
from tileirblas_prefill.api import prefill128 as _prefill128, select_path

_LOCK = Lock()
_COUNTERS = dict(calls=0, launches=0, empty_returns=0, failures=0,
                 prefill128_calls=0, blocked_calls=0)


def stats():
    """Own host counters only; launches mean accepted enqueue, not GPU completion."""
    with _LOCK:
        return dict(_COUNTERS)


def reset_stats():
    """Reset this facade only. Call when no calls are in flight."""
    with _LOCK:
        for key in _COUNTERS:
            _COUNTERS[key] = 0


def selected_path(m, n, k):
    return 'prefill128' if select_path(m, n, k, 32) is not None else 'blocked'


def gemm(A, B, out=None, *, transa=False, transb=True, alpha=1.0, beta=0.0, C=None):
    """BF16 NT facade over frozen cuTile paths; no vendor compute fallback.

    calls counts validated calls, including backend failures. Invalid arguments
    are not counted. Path counters count validated dispatch attempts. Successful
    nonempty returns count launches, successful empty returns empty_returns.
    Backend errors propagate and increment failures. Async CUDA errors occurring
    later cannot be observed by these host counters. No implicit synchronization.
    """
    plan = validate_gemm(A, B, out, transa=transa, transb=transb,
                         alpha=alpha, beta=beta, C=C)
    path = selected_path(plan.m, plan.n, plan.k)
    with _LOCK:
        _COUNTERS['calls'] += 1
        _COUNTERS[path + '_calls'] += 1
    try:
        result = (_prefill128 if path == 'prefill128' else _blocked)(A, B, out=out)
    except BaseException:
        with _LOCK:
            _COUNTERS['failures'] += 1
        raise
    with _LOCK:
        _COUNTERS['launches' if plan.m and plan.n else 'empty_returns'] += 1
    return result


def linear(A, W, out=None):
    return gemm(A, W, out)


def gemm_nt(A, W, out=None):
    return gemm(A, W, out)
