"""CPU-safe contract; no automatic GPU run, model binding, or vendor fallback."""

def contract(m, n, k, mode):
    if any(type(x) is not int for x in (m, n, k)):
        raise TypeError('integer M/N/K required')
    if m not in (32, 1696) or not 0 < n <= 2147483647:
        raise ValueError('initial scope: M32/1696 and positive int32 N')
    if not 0 < k <= 2147483647 or k % 64:
        raise ValueError('positive int32 K divisible by64 required')
    if type(mode) is not str or mode not in ('fp32_partial', 'bf16_partial'):
        raise ValueError('explicit fp32_partial or bf16_partial mode required')
    bm, bn = ((32, 128) if m == 32 else
              ((128, 128) if n >= 1024 else (256, 128)))
    return bm, bn


def tile_ranges(k):
    if type(k) is not int or k <= 0 or k % 64:
        raise ValueError('positive K divisible by64 required')
    half = k // 64
    return range(half), range(half, 2 * half)


def _validate(x, w):
    from tileirblas_blocked.api import validate_gemm
    return validate_gemm(x, w)


def _allocate(x, m, n):
    import torch
    return torch.empty((m, n), device=x.device, dtype=torch.bfloat16)


def _enqueue(x, w, y, k, bm, bn, reduced):
    import torch
    import cuda.tile as ct
    from .kernels import contiguous_split2
    with torch.cuda.device(x.device):
        ct.launch(torch.cuda.current_stream(x.device),
                  ((x.shape[0] + bm - 1)//bm, (w.shape[0] + bn - 1)//bn, 1),
                  contiguous_split2, (x, w, y, k, bm, bn, reduced))


def linear_candidate(x, w, *, mode):
    # Validates the unchanged BF16 NT, alpha1/beta0 tensor/layout/device contract.
    # No precision-setting mutation; no vendor query, fallback, or data-based path.
    plan = _validate(x, w)
    bm, bn = contract(plan.m, plan.n, plan.k, mode)
    y = _allocate(x, plan.m, plan.n)
    _enqueue(x, w, y, plan.k, bm, bn, mode == 'bf16_partial')
    return y
