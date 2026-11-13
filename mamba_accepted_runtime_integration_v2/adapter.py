"""Own public method copy; replace only tensor scatter expression, no global patch."""
from pathlib import Path
import functools,json,hashlib,inspect,textwrap
R=Path(__file__).resolve().parent;ATTR='_accepted_scatter_tileir_v2'
def disjoint_intervals(intervals):
 for i,(a,n)in enumerate(intervals):
  assert n>=0
  for b,m in intervals[i+1:]:assert a+n<=b or b+m<=a,'Overlapping byte intervals rejected'
class Adapter:
 def __init__(self,output):self.output=Path(output).parent;self.failed=False;self.pending=False;self.mode=None;self.calls=0;self.rows=[];self.shadow_ok=False;self.dispatch=0
 def start(self,mode):
  assert not self.failed and not self.pending and mode in ('shadow','replace')
  if mode=='replace':assert self.shadow_ok
  self.mode=mode;self.calls=0;self.rows=[];self.dispatch=0
 def owner(self,t):
  import torch
  assert t.numel()*t.element_size()<=65536 and t.untyped_storage().nbytes()<=65536
  return torch.empty(0,dtype=torch.uint8,device=t.device).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,)).cpu().clone()
 def desc(self,t):return dict(shape=list(t.shape),stride=list(t.stride()),offset=t.storage_offset(),dtype=str(t.dtype),owner_bytes=t.untyped_storage().nbytes())
 def scatter(self,idx,sampled,dst):
  assert not self.failed and self.pending and self.dispatch==self.calls-1
  import torch
  from tileir_vllm_mamba_accepted_scatter_v1.api import scatter_num_accepted_,validate_indices
  assert idx.dtype==torch.int64 and sampled.dtype==dst.dtype==torch.int32 and idx.shape==sampled.shape==dst.shape==(32,)
  assert all(t.stride()==(1,) and t.storage_offset()==0 for t in (idx,sampled,dst))
  assert idx.device==sampled.device==dst.device
  disjoint_intervals([(t.untyped_storage().data_ptr(),t.untyped_storage().nbytes())for t in (idx,sampled,dst)])
  ib=self.owner(idx);sb=self.owner(sampled);before=self.owner(dst);descs=[self.desc(t)for t in (idx,sampled,dst)];ptr=dst.untyped_storage().data_ptr()
  indices=idx.cpu().tolist();values=sampled.cpu().tolist();validate_indices(indices,32)
  expected=before.clone()
  for i,v in zip(indices,values):
   if i>=0:expected.view(torch.int32)[i]=max(v,1)
  if self.mode=='shadow':
   scratch=before.to(dst.device).view(torch.int32);assert scratch.shape==(32,)
   self.native(idx,sampled,dst);ret=scatter_num_accepted_(idx,sampled,scratch);tile=self.owner(scratch)
  else:ret=scatter_num_accepted_(idx,sampled,dst);tile=self.owner(dst)
  actual=self.owner(dst);raw=torch.equal(actual,expected) and torch.equal(tile,expected);preserved=torch.equal(self.owner(idx),ib)and torch.equal(self.owner(sampled),sb);layout=[self.desc(t)for t in (idx,sampled,dst)]==descs and dst.untyped_storage().data_ptr()==ptr
  row=dict(call=self.calls,raw_equal=raw,full_owner_equal=raw,input_preserved=preserved,layout=layout,return_none=ret is None,native_scatter_calls=int(self.mode=='shadow'),tile_scatter_calls=1,descriptors=descs,passed=bool(raw and preserved and layout and ret is None));self.rows.append(row);self.dispatch+=1
  if not row['passed']:
   self.failed=True;path=self.output/self.mode;path.mkdir(parents=True,exist_ok=True);torch.save(dict(indices_owner=ib,sampled_owner=sb,before=before,native_or_actual=actual,tile=tile,expected=expected,row=row),path/'FIRST-FAILURE.pt');raise AssertionError(row)
  return None
 def invoke(self,state,idx,sampled,computed=None):
  assert not self.failed and not self.pending and self.mode in ('shadow','replace')
  self.pending=True;self.calls+=1
  try:
   import torch
   assert isinstance(idx,torch.Tensor) and isinstance(sampled,torch.Tensor),'Only tensor sampled-count path supported'
   assert idx.shape==sampled.shape==(32,),'Only32 actual rows supported'
   assert state.recoverssm is None and not state._align_mode,'Only actual off configuration supported; settings not changed'
   dst=state.num_accepted_tokens_gpu;out=self.copied(state,idx,sampled,computed)
   assert out is None and state.num_accepted_tokens_gpu is dst and self.dispatch==self.calls
   return out
  except BaseException:self.failed=True;raise
  finally:self.pending=False
 def finish(self):
  assert not self.failed and not self.pending and self.calls==self.dispatch==16 and len(self.rows)==16 and all(r['passed']for r in self.rows)
  if self.mode=='shadow':self.shadow_ok=True
  return dict(shadow_pass=True,supported_calls=16,native_scatter_calls=sum(r['native_scatter_calls']for r in self.rows),tile_scatter_calls=16,fallback_calls=0,rows=self.rows)
def install(model,output):
 import vllm.v1.worker.gpu.model_states.mamba_hybrid as m
 assert not hasattr(model,ATTR);a=Adapter(output);a.owner_class=m.MambaHybridModelState;a.original=a.owner_class.postprocess_state
 source=json.loads((R/'SOURCE.json').read_text());assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256'];assert textwrap.dedent(inspect.getsource(a.original)).strip()==(R/'postprocess.original.py.txt').read_text().strip()
 def native(i,s,d):m._scatter_num_accepted_kernel[(32,)](i,s,d)
 a.native=native;ns=dict(a.original.__globals__);ns['_selected_scatter']=a.scatter;exec(compile('from __future__ import annotations\n'+(R/'postprocess.copied.py.txt').read_text(),str(R/'postprocess.copied.py.txt'),'exec'),ns);a.copied=ns['postprocess_state']
 @functools.wraps(a.original)
 def wrapper(state,idx_mapping,num_sampled,num_computed_tokens=None):return a.invoke(state,idx_mapping,num_sampled,num_computed_tokens)
 a.wrapper=wrapper;a.owner_class.postprocess_state=wrapper;setattr(model,ATTR,a);return dict(installed=True,scope='One tensor scatter expression; commit/align original but off-only supported')
def start(model,mode):getattr(model,ATTR).start(mode);return dict(started=True)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 assert a.owner_class.postprocess_state is a.wrapper;a.owner_class.postprocess_state=a.original;delattr(model,ATTR);return dict(restored=True)
