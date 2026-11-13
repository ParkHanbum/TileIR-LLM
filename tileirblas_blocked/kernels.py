# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# See licenses/CUTLASS-BSD-3-Clause.txt and BLOCKED-CANDIDATE.md.
# Public logical CUTLASS adaptation; physical schedule is compiler-owned.
import cuda.tile as ct


@ct.kernel
def blocked_kernel(A, W, Y, M: ct.Constant[int], N: ct.Constant[int], K: ct.Constant[int],
                   BM: ct.Constant[int], BN: ct.Constant[int]):
    # Host dispatch ensures K divisible by32 (including0); nonaligned uses v1.
    acc = ct.full((BM, BN), 0, dtype=ct.float32)
    for tile_id in range(K // 32):
        # Explicit zero padding preserves partial M/N tiles. Row strides come
        # from public Array metadata, including the Qwen padded-A view.
        a32 = ct.load(A, (ct.bid(0), tile_id), (BM, 32), padding_mode=ct.PaddingMode.ZERO)
        w32 = ct.load(W, (ct.bid(1), tile_id), (BN, 32), padding_mode=ct.PaddingMode.ZERO)
        # Subtile indices (0,0)/(0,1), not element offsets. Preserve two K16
        # MMA updates in exactly the logical order of v1, not one K32 MMA.
        a0 = ct.extract(a32, (0, 0), (BM, 16))
        b0 = ct.transpose(ct.extract(w32, (0, 0), (BN, 16)))
        acc = ct.mma(a0, b0, acc)
        a1 = ct.extract(a32, (0, 1), (BM, 16))
        b1 = ct.transpose(ct.extract(w32, (0, 1), (BN, 16)))
        acc = ct.mma(a1, b1, acc)
    result = ct.astype(acc, ct.bfloat16, rounding_mode=ct.RoundingMode.RN)
    ct.store(Y, (ct.bid(0), ct.bid(1)), result)
