"""Source-derived scope, deliberately NOT presented as live observed metadata."""
import math
import struct

VOCAB = 248320
HIDDEN = 2560
BATCH = 32
LENGTHS = (53, 1)
ROPE_DIM = 32
BLOCK = 256


def contiguous_stride(shape):
    out = []
    stride = 1
    for dim in reversed(shape):
        out.append(stride)
        stride *= dim
    return tuple(reversed(out))


def plain_contiguous(meta, shape, dtype, itemsize):
    return (meta is not None and meta['shape'] == tuple(shape)
            and meta['stride'] == contiguous_stride(shape)
            and meta['dtype'] == dtype and meta['offset'] == 0
            and meta['storage_nbytes'] == math.prod(shape) * itemsize)


def embedding_contract(index_meta, weight_meta):
    if not plain_contiguous(weight_meta, (VOCAB, HIDDEN), 'torch.bfloat16', 2):
        return None
    for length in LENGTHS:
        if plain_contiguous(index_meta, (BATCH, length), 'torch.int64', 8):
            return (BATCH, length, HIDDEN)
    return None


def frequency_contract(position_meta, inv_meta):
    if not plain_contiguous(inv_meta, (ROPE_DIM,), 'torch.float32', 4):
        return None
    for length in LENGTHS:
        if plain_contiguous(position_meta, (3, BATCH, length, 1), 'torch.float32', 4):
            return (3, BATCH, length, ROPE_DIM)
    return None


def scale_contract(meta):
    for length in LENGTHS:
        shape = (3, BATCH, length, ROPE_DIM)
        if plain_contiguous(meta, shape, 'torch.float32', 4):
            return shape
    return None


def scalar_bits(value):
    """Finite Python float -> FP32 signed raw word, or unsupported None.

    This is numeric scalar conversion, not an identity specialization. Reject
    overflow/nonfinite scales to native; ordinary finite rounded values and
    signed zero are supported. Raw NaN scalar payload behavior is not claimed.
    """
    if type(value) is not float or not math.isfinite(value):
        return None
    try:
        data = struct.pack('<f', value)
    except OverflowError:
        return None
    return struct.unpack('<i', data)[0]


def disjoint(intervals):
    """Whole backing byte intervals, including distinct-base overlaps."""
    intervals = sorted(intervals)
    return all(a + an <= b for (a, an), (b, _) in zip(intervals, intervals[1:]))


def valid_indices(indices, vocab):
    """Pure CPU specification, never used to inspect live GPU input values."""
    return all(0 <= value < vocab for value in indices)


def embedding_offsets(indices, width):
    return [index * width + column for index in indices for column in range(width)]
