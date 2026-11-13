# SPDX-License-Identifier: Apache-2.0
# Public TileIR scalar store; no compilation/execution evidence yet.
import cuda.tile as ct
@ct.kernel
def fill_one(out, index: ct.Constant[int]):
    ct.scatter(out, ct.full((1,), index, dtype=ct.int32), ct.full((1,), 1, dtype=ct.int32), check_bounds=True)
