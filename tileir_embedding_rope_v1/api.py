"""Narrow inference adapters, not enabled in any model by this package.

Unsupported metadata/options use the original operation once before custom
writes. Exceptions after enqueue attempts propagate without native retry.
Embedding's bounds flag is the one intentional pre-output synchronization.
"""
from dataclasses import dataclass, asdict
from .contracts import (BLOCK, contiguous_stride, disjoint, embedding_contract,
                        frequency_contract, scalar_bits, scale_contract)


@dataclass
class Counters:
    calls: int = 0
    native_fallbacks: int = 0
    bounds_native_fallbacks: int = 0
    allocations: int = 0
    bound_launch_attempts: int = 0
    output_launch_attempts: int = 0
    enqueue_successes: int = 0
    bounds_host_reads: int = 0

    def snapshot(self):
        return asdict(self)


_DEFAULT = {name: Counters() for name in ('embedding', 'frequency_mul', 'scale_mul')}


def stats():
    return {name: counter.snapshot() for name, counter in _DEFAULT.items()}


def reset_stats():
    for name in _DEFAULT:
        _DEFAULT[name] = Counters()


def _counter(counters, site):
    c = _DEFAULT[site] if counters is None else counters
    c.calls += 1
    return c


def _fallback(c, native, *, invalid_bounds=False):
    c.native_fallbacks += 1
    c.bounds_native_fallbacks += int(invalid_bounds)
    return native()


def _context(torch):
    return (torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            and not torch.is_autocast_enabled('cuda'))


def _meta(t, torch, *, weight=False):
    allowed = (torch.Tensor, torch.nn.Parameter) if weight else (torch.Tensor,)
    if (type(t) not in allowed or t.layout != torch.strided
            or t.device.type != 'cuda' or t.device.index != 0
            or t.is_conj() or t.is_neg() or (t.requires_grad and not weight)):
        return None
    return dict(shape=tuple(t.shape), stride=tuple(t.stride()), dtype=str(t.dtype),
                offset=t.storage_offset(), storage_nbytes=t.untyped_storage().nbytes())


def _interval(t):
    storage = t.untyped_storage()
    return storage.data_ptr(), storage.nbytes()


def _disjoint(*tensors):
    return disjoint([_interval(t) for t in tensors])


def _fresh(shape, source, dtype, c, torch):
    out = torch.empty(shape, device=source.device, dtype=dtype)
    c.allocations += 1
    return out


def _output_ok(out, shape, dtype, torch):
    return (type(out) is torch.Tensor and tuple(out.shape) == tuple(shape)
            and tuple(out.stride()) == contiguous_stride(shape)
            and out.dtype is dtype and out.storage_offset() == 0
            and out.untyped_storage().nbytes() == out.numel() * out.element_size()
            and out.is_inference() and not out.requires_grad)


def _kernel(name):
    from . import kernels
    return getattr(kernels, name)


def _launch(device, grid, kernel, args):
    import torch
    import cuda.tile as ct
    with torch.cuda.device(device):
        ct.launch(torch.cuda.current_stream(device), grid, kernel, args)


def _enqueue(c, device, grid, name, args, *, bounds=False):
    # Import/preparation errors precede the enqueue boundary and still propagate.
    kernel = _kernel(name)
    if bounds:
        c.bound_launch_attempts += 1
    else:
        c.output_launch_attempts += 1
    _launch(device, grid, kernel, args)
    c.enqueue_successes += 1


def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False, *, counters=None):
    """F.embedding-compatible inference candidate for source-derived Qwen shapes.

    Padding is a normal row lookup, not forced zeros. Unsupported padding_idx,
    mutation/gradient options, inputs, or aliases use native F.embedding.
    Invalid IDs are NOT clamped, zero-filled, wrapped, or given a fabricated error.
    Their original native call runs before any lookup output is written.
    """
    import torch
    import torch.nn.functional as F
    c = _counter(counters, 'embedding')
    native = lambda: F.embedding(input, weight, padding_idx, max_norm, norm_type,
                                 scale_grad_by_freq, sparse)
    if not _context(torch):
        return _fallback(c, native)
    im, wm = _meta(input, torch), _meta(weight, torch, weight=True)
    shape = embedding_contract(im, wm)
    valid_padding = (padding_idx is None or
                     (type(padding_idx) is int and wm is not None
                      and -wm['shape'][0] <= padding_idx < wm['shape'][0]))
    options = (max_norm is None and type(norm_type) is float and norm_type == 2.0
               and scale_grad_by_freq is False and sparse is False and valid_padding)
    if shape is None or not options or not _disjoint(input, weight):
        return _fallback(c, native)
    flag = _fresh((1,), input, torch.int32, c, torch)
    if not _output_ok(flag, (1,), torch.int32, torch) or not _disjoint(input, weight, flag):
        return _fallback(c, native)
    flat_ids = input.view(-1)
    n_ids, vocab = input.numel(), weight.shape[0]
    bound_block = 1 << (n_ids - 1).bit_length()
    _enqueue(c, input.device, (1, 1, 1), 'index_bounds_kernel',
             (flat_ids, flag, n_ids, vocab, bound_block), bounds=True)
    c.bounds_host_reads += 1
    valid = flag.item()
    if valid == 0:
        return _fallback(c, native, invalid_bounds=True)
    if valid != 1:
        raise RuntimeError('TileIR bounds flag was neither zero nor one')
    out = _fresh(shape, input, weight.dtype, c, torch)
    if (not _output_ok(out, shape, weight.dtype, torch)
            or not _disjoint(input, weight, flag, out)):
        return _fallback(c, native)
    # Source/target raw words preserve every BF16 payload, including sNaN/zero.
    args = (weight.view(-1).view(torch.int16), flat_ids,
            out.view(-1).view(torch.int16), out.numel(), weight.shape[1], BLOCK)
    _enqueue(c, input.device, ((out.numel() + BLOCK - 1) // BLOCK, 1, 1),
             'embedding_raw_kernel', args)
    return out


def rope_frequency_mul(position, inv_freq, *, counters=None):
    """Separate FP32 multiply. No cast/transcendental/recomposition fusion."""
    import torch
    c = _counter(counters, 'frequency_mul')
    native = lambda: position * inv_freq
    shape = (frequency_contract(_meta(position, torch), _meta(inv_freq, torch))
             if _context(torch) else None)
    if shape is None or not _disjoint(position, inv_freq):
        return _fallback(c, native)
    out = _fresh(shape, position, torch.float32, c, torch)
    if (not _output_ok(out, shape, torch.float32, torch)
            or not _disjoint(position, inv_freq, out)):
        return _fallback(c, native)
    _enqueue(c, out.device, ((out.numel() + BLOCK - 1) // BLOCK, 1, 1),
             'rope_frequency_mul_kernel',
             (position.view(-1), inv_freq.view(-1), out.view(-1),
              out.numel(), inv_freq.numel(), BLOCK))
    return out


def rope_scale_mul(tensor, scale, *, counters=None):
    """Fresh FP32 numeric multiply, including scale 1.0. Never returns an alias."""
    import torch
    c = _counter(counters, 'scale_mul')
    native = lambda: tensor * scale
    shape = scale_contract(_meta(tensor, torch)) if _context(torch) else None
    bits = scalar_bits(scale)
    if shape is None or bits is None:
        return _fallback(c, native)
    out = _fresh(shape, tensor, torch.float32, c, torch)
    if (not _output_ok(out, shape, torch.float32, torch)
            or not _disjoint(tensor, out)):
        return _fallback(c, native)
    _enqueue(c, out.device, ((out.numel() + BLOCK - 1) // BLOCK, 1, 1),
             'rope_scale_mul_kernel',
             (tensor.view(-1), out.view(-1), bits, out.numel(), BLOCK))
    return out
