"""CPU-reviewed proposal only. No framework import, launch, or pointer reconstruction.
Integer entries are compared transiently, never used to construct/dereference a Tensor.
Caller must supply freshly verified table entries, NOT guessed owner-derived entries.
"""
from dataclasses import dataclass, field


def require(ok, label):
    if not ok:
        raise ValueError(label)  # no addresses/payload in errors


@dataclass(repr=False)
class ViewLease:
    owner: object = field(repr=False)
    tensor: object = field(repr=False)
    storage: object = field(repr=False)
    shape: tuple
    stride: tuple
    offset: int
    nbytes: int
    itemsize: int
    dtype: str
    device: str

    @classmethod
    def capture(cls, owner):
        t = owner.gpu
        shape, stride = tuple(t.shape), tuple(t.stride())
        require(len(shape) in (1, 2) and all(n > 0 for n in shape), 'shape')
        require(str(t.dtype) in ('torch.int32', 'torch.int64', 'torch.float32'), 'dtype')
        require(str(t.device).startswith('cuda:'), 'device')
        require(stride[-1] == 1 and (len(shape) == 1 or stride[0] == shape[1]), 'dense')
        s = t.untyped_storage()
        lease = cls(owner, t, s, shape, stride, t.storage_offset(), s.nbytes(),
                    t.element_size(), str(t.dtype), str(t.device))
        lease.check()
        return lease

    def check(self):
        t = self.tensor
        require(self.owner.gpu is t, 'owner rebound')
        require(tuple(t.shape) == self.shape and tuple(t.stride()) == self.stride,
                'view changed')
        require(t.storage_offset() == self.offset and self.offset >= 0, 'offset changed')
        require(str(t.dtype) == self.dtype and str(t.device) == self.device, 'type changed')
        s = t.untyped_storage()
        require(s.data_ptr() == self.storage.data_ptr() and s.nbytes() == self.nbytes
                and self.storage.nbytes() == self.nbytes, 'allocation changed')
        require(t.element_size() == self.itemsize, 'element size')
        end = self.offset + 1 + sum((n - 1) * st for n, st in zip(self.shape, self.stride))
        require(end * self.itemsize <= self.nbytes, 'extent')
        require(t.data_ptr() == s.data_ptr() + self.offset * self.itemsize, 'relative offset')
        return t


class FusedLease:
    """Strong typed owners, tables, storage refs; no saved integer addresses.
    No GPU entry acquisition implemented. validate requires fresh actual entries.
    Holding refs does not prevent UVA CPU mutation or pool reuse; external retirement
    barrier remains mandatory before reuse/release. No automatic fallback/retry here.
    """
    def __init__(self, owners, pointer_table, stride_table):
        self.views = tuple(ViewLease.capture(o) for o in owners)
        require(0 < len(self.views) <= 4, 'bounded groups')
        require(all(v.dtype == 'torch.int32' for v in self.views), 'fused int32 only')
        require(len({v.device for v in self.views}) == 1, 'mixed device')
        self.pointer_table, self.stride_table = pointer_table, stride_table

    def validate(self, owners, pointer_table, stride_table, actual_entries, actual_strides):
        require(pointer_table is self.pointer_table and stride_table is self.stride_table,
                'table rebound')
        require(len(owners) == len(self.views) == len(actual_entries) == len(actual_strides),
                'entry count')
        for v, owner, pointer, stride in zip(self.views, owners, actual_entries, actual_strides):
            require(owner is v.owner, 'owner order')
            t = v.check()
            require(type(pointer) is int and pointer == t.data_ptr(), 'entry mismatch')
            require(type(stride) is int and stride == v.stride[0], 'stride mismatch')
        # Conservatively reject shared backing intervals, not just logical overlap.
        for i, a in enumerate(self.views):
            for b in self.views[i + 1:]:
                ap, bp = a.storage.data_ptr(), b.storage.data_ptr()
                require(ap + a.nbytes <= bp or bp + b.nbytes <= ap, 'backing overlap')
        return tuple(v.tensor for v in self.views)


def validate_segments(view, indices, starts, cumulative_lengths):
    """Mock/source contract for non-racing writes. No content values consumed.
    Call on freshly observed host staging metadata before enqueue, not inferred shapes.
    Returns count only, never payload; no tensor creation or mutation.
    """
    view.check()
    require(len(indices) == len(starts) == len(cumulative_lengths), 'metadata lengths')
    rows = view.shape[0]
    width = view.shape[1] if len(view.shape) == 2 else 1
    previous, intervals = 0, []
    for row, start, end in zip(indices, starts, cumulative_lengths):
        require(all(type(x) is int for x in (row, start, end)), 'integer metadata')
        length = end - previous
        require(0 <= row < rows and 0 <= start <= width and 0 <= length <= width - start,
                'write bounds')
        require(end <= 2147483647 and row <= 2147483647, 'int32 metadata')
        lo, hi = row * view.stride[0] + start, row * view.stride[0] + start + length
        require(all(hi <= a or b <= lo or lo == hi or a == b for a, b in intervals),
                'racing writes')
        intervals.append((lo, hi))
        previous = end
    return len(indices)
