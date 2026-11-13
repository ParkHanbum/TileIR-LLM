# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Own TileIR adaptation of public-source blocked_kernel arithmetic.
# Not a claim about proprietary GEMM implementation or schedule.
import cuda.tile as ct

@ct.kernel
def k128_bf16_partials(A,W,Y,K:ct.Constant[int],BM:ct.Constant[int],BN:ct.Constant[int]):
    total=ct.full((BM,BN),0,dtype=ct.float32)
    for part in range(K//128):
        partial=ct.full((BM,BN),0,dtype=ct.float32)
        for j in range(4):
            t=part*4+j
            a=ct.load(A,(ct.bid(0),t),(BM,32),padding_mode=ct.PaddingMode.ZERO)
            w=ct.load(W,(ct.bid(1),t),(BN,32),padding_mode=ct.PaddingMode.ZERO)
            partial=ct.mma(ct.extract(a,(0,0),(BM,16)),ct.transpose(ct.extract(w,(0,0),(BN,16))),partial)
            partial=ct.mma(ct.extract(a,(0,1),(BM,16)),ct.transpose(ct.extract(w,(0,1),(BN,16))),partial)
        rounded=ct.astype(ct.astype(partial,ct.bfloat16,rounding_mode=ct.RoundingMode.RN),ct.float32)
        total=total+rounded
    result=ct.astype(total,ct.bfloat16,rounding_mode=ct.RoundingMode.RN)
    ct.store(Y,(ct.bid(0),ct.bid(1)),result)
