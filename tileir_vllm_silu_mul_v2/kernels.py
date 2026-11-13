# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
# Adapted gather/scatter/cast pattern from TileGym ops/cutile/swiglu.py.
# Changed to packed input, supplied output, 1024-column tiles and x/denominator.
import cuda.tile as ct
from cuda.tile import RoundingMode as RM

@ct.kernel
def silu_mul_kernel(x,out,BLOCK:ct.Constant[int]):
    row=ct.bid(0)
    cols=ct.bid(1)*BLOCK+ct.arange(BLOCK,dtype=ct.int32)
    g=ct.gather(x,(row,cols),check_bounds=False).astype(ct.float32)
    u=ct.gather(x,(row,cols+9216),check_bounds=False).astype(ct.float32)
    denom=1.0+ct.exp(-g)
    activated=ct.truediv(g,denom,rounding_mode=RM.RN).astype(ct.bfloat16).astype(ct.float32)
    y=(activated*u).astype(ct.bfloat16)
    ct.scatter(out,(row,cols),y,check_bounds=False)
