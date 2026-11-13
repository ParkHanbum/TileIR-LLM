"""Original public cuTile API candidates. Not compiled or GPU-qualified.

Every storage-bearing Array argument has a disjoint whole backing. No readonly
alias exceptions, private compiler hooks, or numerical transcendental substitutes.
"""
import cuda.tile as ct


@ct.kernel
def index_bounds_kernel(I, Flag, N: ct.Constant[int], V: ct.Constant[int],
                        B: ct.Constant[int]):
    j = ct.arange(B, dtype=ct.int64)
    ids = ct.gather(I, j, mask=j < N, padding_value=0, check_bounds=True)
    valid = (j >= N) | ((ids >= 0) & (ids < V))
    all_valid = ct.min(ct.astype(valid, ct.int32), axis=0)
    ct.scatter(Flag, 0, all_valid, check_bounds=True)


@ct.kernel
def embedding_raw_kernel(Wbits, I, Obits, N: ct.Constant[int],
                         D: ct.Constant[int], B: ct.Constant[int]):
    # Only launched after the same stream's bounds kernel has completed and its
    # true flag has been read. IDs must not be concurrently mutated by callers.
    j = ct.astype(ct.bid(0), ct.int64) * B + ct.arange(B, dtype=ct.int64)
    ids = ct.gather(I, j // D, mask=j < N, padding_value=0, check_bounds=True)
    bits = ct.gather(Wbits, ids * D + j % D, mask=j < N,
                     padding_value=0, check_bounds=True)
    ct.scatter(Obits, j, bits, mask=j < N, check_bounds=True)


@ct.kernel
def rope_frequency_mul_kernel(P, F, O, N: ct.Constant[int],
                              D: ct.Constant[int], B: ct.Constant[int]):
    j = ct.astype(ct.bid(0), ct.int64) * B + ct.arange(B, dtype=ct.int64)
    pos = ct.gather(P, j // D, mask=j < N, padding_value=0, check_bounds=True)
    inv = ct.gather(F, j % D, mask=j < N, padding_value=0, check_bounds=True)
    val = ct.mul(pos, inv, rounding_mode=ct.RoundingMode.RN, flush_to_zero=False)
    ct.scatter(O, j, val, mask=j < N, check_bounds=True)


@ct.kernel
def rope_scale_mul_kernel(X, O, scale_bits, N: ct.Constant[int], B: ct.Constant[int]):
    j = ct.astype(ct.bid(0), ct.int64) * B + ct.arange(B, dtype=ct.int64)
    x = ct.gather(X, j, mask=j < N, padding_value=0, check_bounds=True)
    # Dynamic INT32 carries a numeric FP32 scalar's bits, not output data. This
    # avoids baking the observed scale 1.0 into code or eliding its multiplication.
    scale = ct.bitcast(ct.full((B,), scale_bits, dtype=ct.int32), ct.float32)
    val = ct.mul(scale, x, rounding_mode=ct.RoundingMode.RN, flush_to_zero=False)
    ct.scatter(O, j, val, mask=j < N, check_bounds=True)
