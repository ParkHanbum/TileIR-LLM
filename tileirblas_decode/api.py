# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# See licenses/CUTLASS-BSD-3-Clause.txt.
from tileirblas_tuned import validate_gemm, gemm_nt as _qualified

CHOICES = ((64,32),(64,64),(256,32),(256,64))
TARGETS = ((32,2560,4096),(32,2560,9216))


def selected_path(m,n,k,bn,bk):
    if (bn,bk) not in CHOICES:
        raise ValueError('candidate geometry is not one of the four declared choices')
    return (32,bn,bk) if (m,n,k) in TARGETS else None


def _run(A,W,out,bn,bk):
    # One public validation precedes all allocation/dispatch. Delegated baseline
    # retains its own validation too; the new path never bypasses real checks.
    plan = validate_gemm(A,W,out)
    if selected_path(plan.m,plan.n,plan.k,bn,bk) is None:
        return _qualified(A,W,out=out)
    import torch
    import cuda.tile as ct
    from .kernels import decode_kernel
    if out is None:
        out = torch.empty((plan.m,plan.n),dtype=torch.bfloat16,device=A.device)
    with torch.cuda.device(A.device):
        ct.launch(torch.cuda.current_stream(A.device),(1,(plan.n+bn-1)//bn,1),
                  decode_kernel,(A,W,out,plan.k,bn,bk))
    return out


def decode64k32(A,W,out=None): return _run(A,W,out,64,32)
def decode64k64(A,W,out=None): return _run(A,W,out,64,64)
def decode256k32(A,W,out=None): return _run(A,W,out,256,32)
def decode256k64(A,W,out=None): return _run(A,W,out,256,64)
