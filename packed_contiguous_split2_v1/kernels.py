# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Logical TileIR adaptation of existing public-source blocked_kernel.
# See CUTLASS-BSD-3-Clause.txt. Not a reconstruction of any vendor schedule.
import cuda.tile as ct


@ct.kernel
def contiguous_split2(A, W, Y, K: ct.Constant[int],
                      BM: ct.Constant[int], BN: ct.Constant[int],
                      ROUND_PARTIALS: ct.Constant[bool]):
    p0 = ct.full((BM, BN), 0, dtype=ct.float32)
    p1 = ct.full((BM, BN), 0, dtype=ct.float32)
    half_tiles = K // 64
    for t in range(half_tiles):
        a = ct.load(A, (ct.bid(0), t), (BM, 32), padding_mode=ct.PaddingMode.ZERO)
        w = ct.load(W, (ct.bid(1), t), (BN, 32), padding_mode=ct.PaddingMode.ZERO)
        p0 = ct.mma(ct.extract(a, (0, 0), (BM, 16)),
                    ct.transpose(ct.extract(w, (0, 0), (BN, 16))), p0)
        p0 = ct.mma(ct.extract(a, (0, 1), (BM, 16)),
                    ct.transpose(ct.extract(w, (0, 1), (BN, 16))), p0)
    for t in range(half_tiles, 2 * half_tiles):
        a = ct.load(A, (ct.bid(0), t), (BM, 32), padding_mode=ct.PaddingMode.ZERO)
        w = ct.load(W, (ct.bid(1), t), (BN, 32), padding_mode=ct.PaddingMode.ZERO)
        p1 = ct.mma(ct.extract(a, (0, 0), (BM, 16)),
                    ct.transpose(ct.extract(w, (0, 0), (BN, 16))), p1)
        p1 = ct.mma(ct.extract(a, (0, 1), (BM, 16)),
                    ct.transpose(ct.extract(w, (0, 1), (BN, 16))), p1)
    if ROUND_PARTIALS:
        p0 = ct.astype(ct.astype(p0, ct.bfloat16, rounding_mode=ct.RoundingMode.RN), ct.float32)
        p1 = ct.astype(ct.astype(p1, ct.bfloat16, rounding_mode=ct.RoundingMode.RN), ct.float32)
    result = ct.astype(p0 + p1, ct.bfloat16, rounding_mode=ct.RoundingMode.RN)
    ct.store(Y, (ct.bid(0), ct.bid(1)), result)
