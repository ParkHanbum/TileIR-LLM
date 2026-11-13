# SPDX-License-Identifier: Apache-2.0
# Public cuTile API implementation, CPU-prepared/uncompiled.
import cuda.tile as ct

@ct.kernel
def zero_bits(out):
    tile = ct.full((1024,), 0, dtype=ct.uint16)
    ct.store(out, (ct.bid(0),), tile)

@ct.kernel
def copy_bits(src, out):
    tile = ct.load(src, (ct.bid(0), 0), (32, 32))
    ct.store(out, (ct.bid(0), 0), tile)
