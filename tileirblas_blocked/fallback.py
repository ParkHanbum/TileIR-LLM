# Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Public CUTLASS-derived logical traversal/epilogue; see licenses/CUTLASS-BSD-3-Clause.txt.
# New cuTile expression, NOT a physical CUTLASS pipeline or cuBLAS reconstruction.
import cuda.tile as ct


@ct.kernel
def nt_kernel(A, W, Y, M: ct.Constant[int], N: ct.Constant[int], K: ct.Constant[int]):
    # Public logical tile 256x128x32. Compiler owns physical warps/layout/stages.
    mi = ct.bid(0) * 256 + ct.arange(256, dtype=ct.int64)
    ni = ct.bid(1) * 128 + ct.arange(128, dtype=ct.int64)
    lane_k = ct.arange(16, dtype=ct.int64)
    acc = ct.full((256, 128), 0, dtype=ct.float32)
    # First residue starts at zero; subsequent full blocks start at residue.
    if K > 0:
        residue = (K - 1) % 32 + 1
        for tile_id in range((K + 31) // 32):
            start = 0
            valid = residue
            if tile_id > 0:
                start = residue + (tile_id - 1) * 32
                valid = 32
            for group in range(2):
                local_k = group * 16 + lane_k
                ki = start + local_k
                a = ct.gather(A, (mi[:, None], ki[None, :]),
                              mask=(mi[:, None] < M) & (local_k[None, :] < valid), padding_value=0)
                b = ct.gather(W, (ni[None, :], ki[:, None]),
                              mask=(ni[None, :] < N) & (local_k[:, None] < valid), padding_value=0)
                acc = ct.mma(a, b, acc)
    # CUTLASS LinearCombination no-source alpha=1 branch: never read Y/C.
    value = ct.astype(acc, ct.bfloat16, rounding_mode=ct.RoundingMode.RN)
    ct.scatter(Y, (mi[:, None], ni[None, :]), value,
               mask=(mi[:, None] < M) & (ni[None, :] < N))
