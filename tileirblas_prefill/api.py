# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Public-source adaptation. See licenses/CUTLASS-BSD-3-Clause.txt.
from tileirblas_blocked import validate_gemm


def select_path(m, n, k, block_k):
    if block_k not in (32, 64):
        raise ValueError('only two prefill candidates are defined')
    if m == 1696 and n >= 1024 and k > 0 and k % block_k == 0:
        return (128, 128, block_k)
    return None


def _run(A, W, out, block_k):
    plan = validate_gemm(A, W, out)
    geometry = select_path(plan.m, plan.n, plan.k, block_k)
    if geometry is None:
        # Preserve validated small-M/small-N/residue/empty dispatch exactly.
        from tileirblas_blocked import gemm_nt as baseline
        return baseline(A, W, out=out)
    import torch
    import cuda.tile as ct
    from .kernels import prefill_kernel
    if out is None:
        out = torch.empty((plan.m, plan.n), dtype=torch.bfloat16, device=A.device)
    with torch.cuda.device(A.device):
        ct.launch(torch.cuda.current_stream(A.device), ((plan.m+127)//128, (plan.n+127)//128, 1),
                  prefill_kernel, (A, W, out, plan.k, block_k))
    return out


def prefill128(A, W, out=None):
    """128x128x32 load tiles, ordered K16 MMA groups."""
    return _run(A, W, out, 32)


def prefill128k64(A, W, out=None):
    """128x128x64 load tiles, ordered K16 MMA groups."""
    return _run(A, W, out, 64)


def gemm_nt(A, W, out=None):
    return prefill128(A, W, out)
