"""Typed destination API; no pointer tables, owner allocation or staging-list mutation."""
from .contract import validate
def apply(out,indices,starts,contents,ends,*,ledger=None):
 validate(tuple(out.shape),tuple(out.stride()),str(out.dtype),indices,starts,contents,ends)
 assert out.device.type=='cuda'and not out.requires_grad and out.storage_offset()==0
 assert out.numel()*out.element_size()==out.untyped_storage().nbytes()<=67108864
 entry=dict(nonempty=bool(indices),launch_attempts=0,launch_returns=0,scope='source ct.launch boundary; not physical GPU count')
 if ledger is not None:ledger.append(entry)
 if not indices:return None
 import torch,cuda.tile as ct
 from .kernels import write_segments
 # Same public destination dtype conversion as native async_tensor_h2d.
 i=torch.tensor(indices,dtype=torch.int32,device=out.device)
 s=torch.tensor(starts,dtype=torch.int32,device=out.device)
 e=torch.tensor(ends,dtype=torch.int32,device=out.device)
 c=torch.tensor(contents,dtype=out.dtype,device=out.device)
 stream=torch.cuda.current_stream(out.device)
 entry['launch_attempts']+=1
 ct.launch(stream,(len(indices),),write_segments,(out.view(-1),i,s,c,e,out.stride(0)))
 entry['launch_returns']+=1
 # Diagnostic API requires retirement before temporary metadata is released.
 stream.synchronize()
 return None
def fused(outputs,snapshots,*,ledger=None):
 assert 0<len(outputs)<=4 and len(outputs)==len(snapshots)
 assert all(str(t.dtype)=='torch.int32'and t.device==outputs[0].device for t in outputs)
 for t,data in zip(outputs,snapshots):validate(tuple(t.shape),tuple(t.stride()),str(t.dtype),*data)
 spans=[(t.untyped_storage().data_ptr(),t.untyped_storage().nbytes())for t in outputs]
 assert all(a+n<=b or b+m<=a for i,(a,n)in enumerate(spans)for b,m in spans[i+1:])
 for t,data in zip(outputs,snapshots):apply(t,*data,ledger=ledger)
 return None
