# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Public logical CUTLASS adaptation; no physical pipeline identity claimed.
import cuda.tile as ct


@ct.kernel
def prefill_kernel(A, W, Y, K: ct.Constant[int], BK: ct.Constant[int]):
    acc = ct.full((128, 128), 0, dtype=ct.float32)
    for tile_id in range(K // BK):
        av = ct.load(A, (ct.bid(0), tile_id), (128, BK), padding_mode=ct.PaddingMode.ZERO)
        wv = ct.load(W, (ct.bid(1), tile_id), (128, BK), padding_mode=ct.PaddingMode.ZERO)
        a0 = ct.extract(av, (0, 0), (128, 16))
        b0 = ct.transpose(ct.extract(wv, (0, 0), (128, 16)))
        acc = ct.mma(a0, b0, acc)
        a1 = ct.extract(av, (0, 1), (128, 16))
        b1 = ct.transpose(ct.extract(wv, (0, 1), (128, 16)))
        acc = ct.mma(a1, b1, acc)
        if BK == 64:
            a2 = ct.extract(av, (0, 2), (128, 16))
            b2 = ct.transpose(ct.extract(wv, (0, 2), (128, 16)))
            acc = ct.mma(a2, b2, acc)
            a3 = ct.extract(av, (0, 3), (128, 16))
            b3 = ct.transpose(ct.extract(wv, (0, 3), (128, 16)))
            acc = ct.mma(a3, b3, acc)
    ct.store(Y, (ct.bid(0), ct.bid(1)), ct.astype(acc, ct.bfloat16, rounding_mode=ct.RoundingMode.RN))
