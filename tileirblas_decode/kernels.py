# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Public logical adaptation, not physical CUTLASS or cuBLAS kernel identity.
import cuda.tile as ct


@ct.kernel
def decode_kernel(A,W,Y,K:ct.Constant[int],BN:ct.Constant[int],BK:ct.Constant[int]):
    acc = ct.full((32,BN),0,dtype=ct.float32)
    for tid in range(K//BK):
        av = ct.load(A,(0,tid),(32,BK),padding_mode=ct.PaddingMode.ZERO)
        wv = ct.load(W,(ct.bid(1),tid),(BN,BK),padding_mode=ct.PaddingMode.ZERO)
        a0 = ct.extract(av,(0,0),(32,16))
        b0 = ct.transpose(ct.extract(wv,(0,0),(BN,16)))
        acc = ct.mma(a0,b0,acc)
        a1 = ct.extract(av,(0,1),(32,16))
        b1 = ct.transpose(ct.extract(wv,(0,1),(BN,16)))
        acc = ct.mma(a1,b1,acc)
        if BK == 64:
            a2 = ct.extract(av,(0,2),(32,16))
            b2 = ct.transpose(ct.extract(wv,(0,2),(BN,16)))
            acc = ct.mma(a2,b2,acc)
            a3 = ct.extract(av,(0,3),(32,16))
            b3 = ct.transpose(ct.extract(wv,(0,3),(BN,16)))
            acc = ct.mma(a3,b3,acc)
    ct.store(Y,(0,ct.bid(1)),ct.astype(acc,ct.bfloat16,rounding_mode=ct.RoundingMode.RN))
