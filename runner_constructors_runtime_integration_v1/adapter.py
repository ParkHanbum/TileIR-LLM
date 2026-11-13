"""Owned public method copy; only two constructor expressions replaced. No global Torch patch."""
import json,hashlib,functools,inspect,textwrap
from pathlib import Path
R=Path(__file__).resolve().parent
ATTR='_runner_constructors_tileir_v1'
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.failed=False;self.mode=None;self.pending=False;self.rows=[];self.calls=0;self.current=[];self.kept=[];self.shadow_ok=False
 def check(self):assert not self.failed,'Permanent first-failure latch'
 def start(self,mode):
  self.check();assert not self.pending and mode in ('shadow','replace')
  if mode=='replace':assert self.shadow_ok
  self.mode=mode;self.rows=[];self.calls=0;self.kept=[]
 def construct(self,kind,n,device):
  self.check();assert self.pending and len(self.current)<2
  import torch
  from tileir_vllm_runner_constructors_v1 import api
  assert kind==('cu_num_logits','expanded_local_pos')[len(self.current)] and type(n)is int and n==32
  expected=torch.arange(33,dtype=torch.int32) if kind=='cu_num_logits' else torch.zeros(32,dtype=torch.int32)
  native=None
  if self.mode=='shadow':native=torch.arange(33,dtype=torch.int32,device=device) if kind=='cu_num_logits' else torch.zeros(32,dtype=torch.int32,device=device)
  tile=getattr(api,kind)(n,device);cpu=tile.detach().cpu()
  def layout(t):return t.dtype==torch.int32 and tuple(t.shape)==tuple(expected.shape) and t.stride()==(1,) and t.storage_offset()==0 and t.untyped_storage().nbytes()==expected.numel()*4 and t.device==torch.device(device)
  passed=layout(tile) and torch.equal(cpu,expected)
  if native is not None:passed=passed and layout(native) and torch.equal(native.detach().cpu(),expected) and native.untyped_storage().data_ptr()!=tile.untyped_storage().data_ptr()
  prior=all(torch.equal(t.detach().cpu(),e) for t,e in self.kept)
  fresh=all(tile.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr() and (native is None or native.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr()) for t,e in self.kept)
  row=dict(call=self.calls,kind=kind,raw_equal=bool(passed),prior_unchanged=prior,fresh_disjoint=fresh,native_calls=int(native is not None),tile_calls=1,passed=bool(passed and prior and fresh))
  self.rows.append(row)
  if not row['passed']:
   self.failed=True;self.output.mkdir(parents=True,exist_ok=True);torch.save(dict(expected=expected,tile=cpu,native=None if native is None else native.detach().cpu()),self.output/'first-failure.pt');raise AssertionError(row)
  out=native if native is not None else tile
  self.kept.extend([(tile,expected.clone())]+([] if native is None else [(native,expected.clone())]));self.current.append(out);return out
 def invoke(self,copied,runner,*a,**k):
  self.check();assert not self.pending and self.mode in ('shadow','replace')
  self.pending=True;self.current=[];self.calls+=1
  try:
   bound=inspect.signature(self.original).bind(runner,*a,**k);bound.apply_defaults();s=bound.arguments['scheduler_output'];b=bound.arguments['batch_req_state']
   assert not s.scheduled_spec_decode_tokens and len(b.req_ids)==32 and runner.model_state.num_new_sampled_tokens_per_step==1 and runner.pcp_manager is None
   out=copied(runner,*a,**k)
   assert len(self.current)==2 and out.cu_num_logits is self.current[0] and out.expanded_local_pos is self.current[1]
   assert out.req_ids is b.req_ids and out.idx_mapping is out.expanded_idx_mapping and out.num_reqs==32 and out.num_draft_tokens==0
   return out
  except BaseException:self.failed=True;raise
  finally:self.pending=False
 def finish(self):
  self.check();assert not self.pending and self.calls==16 and len(self.rows)==32 and all(x['passed'] for x in self.rows)
  if self.mode=='shadow':self.shadow_ok=True
  return dict(shadow_pass=True,supported_calls=16,logical_suboperation_calls=32,native_constructor_calls=sum(x['native_calls'] for x in self.rows),tileir_constructor_calls=32,supported_vendor_fallback_calls=0,rows=self.rows)
def install(model,output):
 import vllm.v1.worker.gpu.model_runner as mr
 assert not hasattr(model,ATTR)
 a=Adapter(output);old=mr.GPUModelRunner.prepare_inputs;a.original=old
 pin=json.loads((R/'SOURCE.json').read_text());assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
 assert textwrap.dedent(inspect.getsource(old)).strip()==(R/'prepare_inputs.original.py.txt').read_text().strip()
 ns=dict(old.__globals__);ns['_construct']=a.construct;exec(compile('from __future__ import annotations\n'+(R/'prepare_inputs.copied.py.txt').read_text(),str(R/'prepare_inputs.copied.py.txt'),'exec'),ns);copied=ns['prepare_inputs']
 @functools.wraps(old)
 def wrapped(runner,*args,**kwargs):return a.invoke(copied,runner,*args,**kwargs)
 a.owner=mr.GPUModelRunner;a.wrapper=wrapped;mr.GPUModelRunner.prepare_inputs=wrapped;setattr(model,ATTR,a)
 return dict(installed=True,scope='Only two owned constructor expressions; original side effects execute once')
def start(model,mode):getattr(model,ATTR).start(mode);return dict(started=True)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 assert a.owner.prepare_inputs is a.wrapper
 a.owner.prepare_inputs=a.original;delattr(model,ATTR);return dict(restored=True)
