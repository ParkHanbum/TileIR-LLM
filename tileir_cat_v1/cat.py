"""Site-bound experimental cat adapter. No registration or CUDA on import."""
from dataclasses import dataclass, field, asdict
from .contract import BLOCK, Snapshot, UnsupportedCat, match_contract, validate_output

@dataclass
class Counters:
    calls: int = 0
    allocations: int = 0
    allocation_failures: int = 0
    native_fallbacks: int = 0
    native_failures: int = 0
    launch_attempts: int = 0
    launch_failures: int = 0
    tileir_cats: int = 0
    fallback_reasons: dict = field(default_factory=dict)
    def snapshot(self): return asdict(self)

_DEFAULT = Counters()

def stats(): return _DEFAULT.snapshot()
def reset_stats():
    global _DEFAULT
    _DEFAULT = Counters()


def _tensor_snapshot(t, *, input=True):
    import torch
    if type(t) is not torch.Tensor:
        raise UnsupportedCat('not_plain_tensor')
    if t.layout != torch.strided:
        raise UnsupportedCat('not_strided')
    if t.device.type != 'cuda':
        raise UnsupportedCat('not_cuda')
    if not torch.is_inference_mode_enabled() or torch.is_grad_enabled() or not t.is_inference():
        raise UnsupportedCat('not_observed_inference_mode')
    if t.requires_grad or t.is_conj() or t.is_neg():
        raise UnsupportedCat('grad_conj_or_neg')
    storage = t.untyped_storage()
    m = dict(shape=list(t.shape), stride=list(t.stride()), storage_offset=t.storage_offset(),
        storage_nbytes=storage.nbytes(), dtype=str(t.dtype), device=str(t.device),
        element_size=t.element_size(), layout=str(t.layout), requires_grad=t.requires_grad,
        is_contiguous=t.is_contiguous(), is_inference=t.is_inference(), is_conj=t.is_conj(),
        is_neg=t.is_neg(), is_subclass=False, tensor_type='torch.Tensor', version=None,
        grad_enabled=torch.is_grad_enabled(), inference_mode_enabled=torch.is_inference_mode_enabled())
    if input: m['kind'] = 'tensor'
    return Snapshot(m, id(t), storage.data_ptr())


def _allocate(inputs, plan):
    import torch
    return torch.empty(tuple(plan.output['shape']), dtype=inputs[0].dtype,
                       device=inputs[0].device, memory_format=torch.contiguous_format)


def _launch_prepared(inputs, output, plan):
    import torch
    import cuda.tile as ct
    from . import kernels as k
    n = plan.numel
    grid = ((n + BLOCK - 1)//BLOCK,1,1)
    family = plan.kernel_family
    if family == 'gdn32':
        t = plan.output['shape'][2]
        grid = ((8192*t + BLOCK - 1)//BLOCK,32,1)
        kernel, args = k.gdn32_kernel, (tuple(inputs),output,t,BLOCK)
    elif family == 'kv_initial':
        kernel,args = k.kv_initial_kernel,(inputs[1],output,53,n,BLOCK)
    elif family == 'kv_append':
        kernel,args = k.kv_append_kernel,(inputs[0],inputs[1],output,plan.output['shape'][2]-1,n,BLOCK)
    elif family == 'rope':
        kernel,args = k.rope_kernel,(inputs[0],output,plan.output['shape'][1],n,BLOCK)
    elif family == 'initial_positions':
        kernel,args = k.initial_positions_kernel,(inputs[0],output,n,BLOCK)
    elif family == 'positions':
        kernel,args = k.positions_kernel,(inputs[0],inputs[1],output,plan.output['shape'][2]-1,n,BLOCK)
    else:
        raise AssertionError('unknown admitted kernel family')
    with torch.cuda.device(output.device):
        ct.launch(torch.cuda.current_stream(output.device),grid,kernel,args)


def cat_observed(tensors, dim=0, *, site, out=None, counters=None):
    """Native torch.cat fallback for anything outside the exact observed site.

    Launch/compile failures propagate, even when no write is known to occur.
    No retry/native fallback is attempted after custom launch begins. Every
    successful custom path returns a newly allocated compact inference tensor.
    CPU input always takes native fallback; no GPU is queried for admission.
    """
    import torch
    c = _DEFAULT if counters is None else counters
    c.calls += 1
    def fallback(reason):
        c.native_fallbacks += 1
        c.fallback_reasons[reason] = c.fallback_reasons.get(reason,0)+1
        try:
            if out is None: return torch.cat(tensors,dim=dim)
            return torch.cat(tensors,dim=dim,out=out)
        except BaseException:
            c.native_failures += 1
            raise
    try:
        if type(tensors) not in (list,tuple):
            raise UnsupportedCat('not_plain_tensor_sequence')
        if type(dim) is not int or out is not None:
            raise UnsupportedCat('dim_type_or_out')
        snapshots = tuple(_tensor_snapshot(t) for t in tensors)
        plan = match_contract(site,snapshots,dim,out_is_none=out is None)
    except UnsupportedCat as error:
        return fallback(str(error))
    try:
        output = _allocate(tensors,plan)
    except BaseException:
        c.allocation_failures += 1
        raise
    c.allocations += 1
    try:
        validate_output(plan,_tensor_snapshot(output,input=False),snapshots)
    except UnsupportedCat as error:
        return fallback(str(error))
    c.launch_attempts += 1
    try:
        _launch_prepared(tensors,output,plan)
    except BaseException:
        c.launch_failures += 1
        raise
    c.tileir_cats += 1
    return output
