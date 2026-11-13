# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Derived K traversal: full license in licenses/CUTLASS-BSD-3-Clause.txt.
"""Host contract; importing this module does not import torch or initialize CUDA."""
from dataclasses import dataclass

_COUNTERS = dict(calls=0, launches=0, empty_returns=0)

@dataclass(frozen=True)
class GemmPlan:
    m: int
    n: int
    k: int
    a_row_stride: int
    w_row_stride: int
    out_row_stride: int
    tile: tuple = (256, 128, 32)


def stats():
    return dict(_COUNTERS)


def reset_stats():
    for key in _COUNTERS:
        _COUNTERS[key] = 0


def k_tiles(k, block=32):
    """Public CUTLASS logical first-residue order (offset, valid count)."""
    if not isinstance(k, int) or k < 0 or block <= 0:
        raise ValueError('nonnegative integer K and positive block required')
    if not k:
        return ()
    residue = (k - 1) % block + 1
    return ((0, residue),) + tuple((i, block) for i in range(residue, k, block))


def _matrix(x, name, require_cuda):
    import torch
    if not isinstance(x, torch.Tensor):
        raise TypeError(name + ' must be torch.Tensor')
    if x.layout != torch.strided or x.ndim != 2 or x.dtype != torch.bfloat16:
        raise ValueError(name + ' must be 2D strided BF16')
    if require_cuda and x.device.type != 'cuda':
        raise ValueError(name + ' must be CUDA')
    if x.requires_grad:
        raise ValueError('autograd is unsupported; supply detached tensors')
    if x.stride(1) != 1 or x.stride(0) < x.shape[1]:
        raise ValueError(name + ' requires unit inner stride and row stride >= width')
    if any(d > 2147483647 for d in x.shape):
        raise ValueError('dimensions exceed int32 API range')


def validate_gemm(A, B, out=None, *, transa=False, transb=True,
                  alpha=1.0, beta=0.0, C=None, require_cuda=True):
    """Validate without allocation, mutation, CUDA initialization, or compilation.

    Only real BF16 NT, alpha=1, beta=0, no C/bias/activation is supported.
    Logical B is the transpose of physical row-major W[N,K].
    Output must be contiguous. Storage aliasing with either input is rejected.
    """
    if transa is not False or transb is not True:
        raise NotImplementedError('only transa=False, transb=True is supported')
    if isinstance(alpha, bool) or isinstance(beta, bool) or not isinstance(alpha, (int, float)) or not isinstance(beta, (int, float)):
        raise TypeError('alpha and beta must be real host scalars')
    if alpha != 1.0 or beta != 0.0 or C is not None:
        raise NotImplementedError('only alpha=1, beta=0, C=None is supported')
    _matrix(A, 'A', require_cuda)
    _matrix(B, 'W', require_cuda)
    if A.device != B.device or A.shape[1] != B.shape[1]:
        raise ValueError('input devices and K dimensions must match')
    m, k = A.shape
    n = B.shape[0]
    if out is not None:
        _matrix(out, 'out', require_cuda)
        if out.device != A.device or tuple(out.shape) != (m, n) or not out.is_contiguous():
            raise ValueError('out must be contiguous BF16 [M,N] on input device')
        if out.numel() and any(out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr() for x in (A, B)):
            raise ValueError('out must not share storage with inputs')
    return GemmPlan(m, n, k, A.stride(0), B.stride(0), n)


def gemm(A, B, out=None, *, transa=False, transb=True, alpha=1.0, beta=0.0, C=None, tile_policy="shape"):
    """Return out = A @ B.T, enqueued on current PyTorch CUDA stream.

    No implicit dtype casts, input copies, synchronization, autograd or BLAS
    fallback. K=0 writes +0 using cuTile; M=0 or N=0 launches nothing.
    Validation always precedes allocation/mutation. CPU tensors are never run.
    """
    plan = validate_gemm(A, B, out, transa=transa, transb=transb, alpha=alpha, beta=beta, C=C)
    geometry = select_tile(plan.m, plan.n, tile_policy)
    import torch
    if out is None:
        out = torch.empty((plan.m, plan.n), dtype=torch.bfloat16, device=A.device)
    _COUNTERS['calls'] += 1
    if not plan.m or not plan.n:
        _COUNTERS['empty_returns'] += 1
        return out
    import cuda.tile as ct
    if plan.k % 32:
        # Frozen v1 exact masked first-residue fallback, no change in geometry.
        from .fallback import nt_kernel
        bm, bn = 256, 128
        args = (A, B, out, plan.m, plan.n, plan.k)
    else:
        from .kernels import blocked_kernel as nt_kernel
        bm, bn = geometry
        args = (A, B, out, plan.m, plan.n, plan.k, bm, bn)
    with torch.cuda.device(A.device):
        ct.launch(torch.cuda.current_stream(A.device), ((plan.m + bm-1)//bm, (plan.n + bn-1)//bn, 1),
                  nt_kernel, args)
    _COUNTERS['launches'] += 1
    return out


def linear(A, W, out=None):
    """Bias-free BF16 linear adapter; physical W is [N,K]."""
    return gemm(A, W, out)


def gemm_nt(A, W, out=None):
    return linear(A, W, out)


def select_tile(m, n, policy="shape"):
    """Only two comparison policies; no autotuning or device query."""
    if policy == "fixed":
        return (256, 128)
    if policy != "shape":
        raise ValueError("tile_policy must be shape or fixed")
    if m <= 32:
        return (32, 128)
    if n <= 32:
        return (32, 32)
    return (256, 128)


def gemm_nt_fixed(A, W, out=None):
    """Fixed public-fixture geometry control for paired comparison."""
    return gemm(A, W, out, tile_policy="fixed")
