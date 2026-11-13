# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# See licenses/CUTLASS-BSD-3-Clause.txt. Exact frozen kernels; no new math.
from threading import Lock
from tileirblas_blocked import validate_gemm
from ._enqueue import enqueue as _enqueue

_LOCK = Lock()
_COUNTERS = dict(calls=0, launches=0, empty_returns=0, failures=0,
                 decode64k64_calls=0, prefill128_calls=0, blocked_calls=0,
                 residue_calls=0, empty_calls=0)


def selected_path(m,n,k):
    if not m or not n:return 'empty'
    if (m,n,k) in ((32,2560,4096),(32,2560,9216)):return 'decode64k64'
    if m==1696 and n>=1024 and k>0 and k%32==0:return 'prefill128'
    if k%32:return 'residue'
    return 'blocked'


def stats():
    with _LOCK:return dict(_COUNTERS)


def reset_stats():
    """Reset facade-only diagnostics with no calls in flight."""
    with _LOCK:
        for key in _COUNTERS:_COUNTERS[key]=0


def gemm(A,B,out=None,*,transa=False,transb=True,alpha=1.0,beta=0.0,C=None):
    """Validate once; use a private enqueue plan. No public unchecked entry point.

    Validated attempts count as calls/path_calls, synchronous failures propagate.
    launches counts successful host enqueues, not device completion. Invalid
    arguments do not count. No implicit synchronization or vendor fallback.
    """
    plan=validate_gemm(A,B,out,transa=transa,transb=transb,alpha=alpha,beta=beta,C=C)
    path=selected_path(plan.m,plan.n,plan.k)
    with _LOCK:
        _COUNTERS['calls']+=1
        _COUNTERS[path+'_calls']+=1
    try:
        result=_enqueue(A,B,out,plan,path)
    except BaseException:
        with _LOCK:_COUNTERS['failures']+=1
        raise
    with _LOCK:_COUNTERS['empty_returns' if path=='empty' else 'launches']+=1
    return result


def linear(A,W,out=None):return gemm(A,W,out)
def gemm_nt(A,W,out=None):return gemm(A,W,out)
