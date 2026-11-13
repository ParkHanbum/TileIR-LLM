"""Actual observed metadata restriction; no torch/CUDA import at module scope."""
import json
from pathlib import Path
NAMES=('idx_mapping','num_computed_tokens','last_sampled_tokens','output_bin_counts','sampled_tokens','num_sampled','num_rejected','query_start_loc','all_token_ids','total_len')
WRITES=(1,2,3,8,9)
READS=(0,4,5,6,7)
EXPECTED=json.loads((Path(__file__).parent/'OBSERVED.json').read_text())['parameters']

def validate_metadata(args):
    if len(args)!=10:raise ValueError('Exactly10 required tensors')
    intervals=[]
    for name,x in zip(NAMES,args):
        e=EXPECTED[name]
        if x is None:raise ValueError('Optional-None path not observed/qualified')
        if (list(x.shape)!=e['shape'] or list(x.stride())!=e['stride'] or str(x.dtype)!=e['dtype'] or str(x.device)!='cuda:0' or str(x.layout)!='torch.strided' or x.storage_offset()!=0 or x.requires_grad or x.is_conj() or x.is_neg()):raise ValueError('Unsupported metadata: '+name)
        storage=x.untyped_storage();start=storage.data_ptr();size=storage.nbytes()
        if size!=e['storage_bytes'] or start<=0:raise ValueError('Full exact-size backing required: '+name)
        intervals.append((start,start+size,name))
    for i,a in enumerate(intervals):
        for b in intervals[i+1:]:
            if a[0]<b[1] and b[0]<a[1]:raise ValueError('Overlapping full inputs: '+a[2]+','+b[2])

def clone_inputs(torch,args):
    # Retained device allocations explicitly change UVA residency. This is a
    # diagnostic arithmetic gate, not proof of candidate on original UVA views.
    validate_metadata(args)
    clones=tuple(x.clone(memory_format=torch.contiguous_format) for x in args)
    validate_metadata(clones)
    return clones

def validate_values(torch,args):
    # Local booleans only, no index/token/state payload returned or persisted.
    idx,computed,last,hist,sampled,ns,nr,q,tokens,total=args
    active=idx>=0; states=idx[active]
    def need(v,message):
        if not bool(v):raise ValueError(message)
    need((states<32).all(),'state index bounds')
    need(torch.unique(states).numel()==states.numel(),'conflicting active state indices')
    n=ns[active].to(torch.int64); old=total[states].to(torch.int64)
    need(((n>=0)&(n<=1)).all(),'sample count exceeds observed row')
    need(((old>=0)&(old+n<=128)).all(),'append bounds')
    accepted=active & (ns>0); token=sampled[:,0][accepted]; dest=idx[accepted]
    need(((token>=0)&(token<248320)).all(),'histogram/token bounds')
    need((hist[dest,token].to(torch.int64)<2147483647).all(),'histogram integer overflow')
    query_delta=q[1:].to(torch.int64)-q[:-1].to(torch.int64)
    delta=query_delta-nr.to(torch.int64)
    def fits(v):return ((v>=-2147483648)&(v<=2147483647)).all()
    need(fits(query_delta[active]) and fits(delta[active]),'query/rejection arithmetic overflow')
    need(fits(computed[states].to(torch.int64)+delta[active]),'computed integer overflow')
