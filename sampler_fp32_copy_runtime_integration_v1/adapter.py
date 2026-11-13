"""One owned-copy expression, subsequent processors and sampler unchanged."""
import json,hashlib,inspect,textwrap,functools
from pathlib import Path
R=Path(__file__).resolve().parent;ATTR='_sampler_fp32_copy_tileir_v1'
class Adapter:
 def __init__(self,output):self.output=Path(output);self.failed=False;self.mode=None;self.pending=False;self.rows=[];self.calls=0;self.current=None;self.shadow_ok=False;self.sampler=None
 def check(self):assert not self.failed,'Permanent first-failure latch'
 def start(self,mode):
  self.check();assert not self.pending and mode in ('shadow','replace')
  if mode=='replace':assert self.shadow_ok
  self.mode=mode;self.rows=[];self.calls=0
 def copy_logits(self,x):
  self.check();assert self.pending and self.current is None
  import torch
  from tileir_vllm_sampler_fp32_copy_v1 import sampler_fp32_copy
  from tileir_vllm_sampler_fp32_copy_v1.api import validate_metadata
  validate_metadata(dict(shape=tuple(x.shape),stride=tuple(x.stride()),offset=x.storage_offset(),dtype=str(x.dtype)))
  assert x.untyped_storage().nbytes()==15892480 and not x.requires_grad
  assert bool(torch.isfinite(x).all()),'Finite workload scope only'
  def digest(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
  before=digest(x);native=torch.empty_like(x,dtype=torch.float32).copy_(x) if self.mode=='shadow' else None;tile=sampler_fp32_copy(x)
  def layout(y):return y.shape==x.shape and y.dtype==torch.float32 and y.device==x.device and y.stride()==(248320,1) and y.storage_offset()==0 and y.untyped_storage().nbytes()==31784960
  fresh=tile.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr() and (native is None or all(native.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr()for t in (x,tile)))
  raw=None if native is None else torch.equal(tile.view(torch.int32),native.view(torch.int32));strict=None if native is None else bool(torch.allclose(tile,native,atol=1e-5,rtol=1e-5));finite=bool(torch.isfinite(tile).all());unchanged=before==digest(x)
  row=dict(call=self.calls,native_calls=int(native is not None),tile_calls=1,raw_equal=raw,strict=strict,finite=finite,layout=layout(tile)and(native is None or layout(native)),fresh=fresh,input_unchanged=unchanged)
  row['passed']=all(row[k]for k in ('finite','layout','fresh','input_unchanged'))and(native is None or(raw and strict));self.rows.append(row)
  if not row['passed']:
   self.failed=True;self.output.mkdir(parents=True,exist_ok=True);torch.save(dict(x=x.cpu(),tile=tile.cpu(),native=None if native is None else native.cpu()),self.output/'first-failure.pt');raise AssertionError(row)
  self.current=native if native is not None else tile;self.after_check=lambda:before==digest(x);self.immediate_hash=digest(self.current);self.digest=digest
  return self.current
 def invoke(self,copied,owner,*a,**kw):
  self.check();assert not self.pending and self.mode in ('shadow','replace');self.pending=True;self.current=None;self.calls+=1
  try:
   if self.sampler is None:self.sampler=owner
   assert owner is self.sampler and self.calls<=16
   ret=copied(owner,*a,**kw);assert self.current is not None
   row=self.rows[-1];row['post_method_inputs_preserved']=self.after_check();assert row['post_method_inputs_preserved']
   row['method_return_same_copy']=ret is self.current;assert row['method_return_same_copy'],'Captured skip-top-k/p branch returns same buffer'
   row['post_processor_buffer_changed']=self.digest(ret)!=self.immediate_hash
   return ret
  except BaseException:self.failed=True;raise
  finally:self.pending=False
 def finish(self):
  self.check();assert not self.pending and self.calls==16 and len(self.rows)==16 and all(x['passed']for x in self.rows)
  if self.mode=='shadow':assert all(x['raw_equal'] and x['strict']for x in self.rows);self.shadow_ok=True
  return dict(shadow_pass=self.shadow_ok,supported_calls=16,native_copy_calls=sum(x['native_calls']for x in self.rows),tileir_copy_calls=16,supported_vendor_fallback_calls=0,rows=self.rows)
def install(model,output):
 from vllm.v1.worker.gpu.sample.sampler import Sampler
 assert not hasattr(model,ATTR);a=Adapter(output);old=Sampler.apply_sampling_params
 pin=json.loads((R/'SOURCE.json').read_text());assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
 assert textwrap.dedent(inspect.getsource(old)).strip()==(R/'method.original.py.txt').read_text().strip()
 ns=dict(old.__globals__);ns['_copy_logits']=a.copy_logits;exec(compile('from __future__ import annotations\n'+(R/'method.copied.py.txt').read_text(),str(R/'method.copied.py.txt'),'exec'),ns);copied=ns['apply_sampling_params']
 @functools.wraps(old)
 def wrapped(owner,*args,**kwargs):return a.invoke(copied,owner,*args,**kwargs)
 a.owner=Sampler;a.original=old;a.wrapper=wrapped;Sampler.apply_sampling_params=wrapped;setattr(model,ATTR,a);return dict(installed=True)
def start(model,mode):getattr(model,ATTR).start(mode);return dict(started=True)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 assert a.owner.apply_sampling_params is a.wrapper;a.owner.apply_sampling_params=a.original;delattr(model,ATTR);return dict(restored=True)
