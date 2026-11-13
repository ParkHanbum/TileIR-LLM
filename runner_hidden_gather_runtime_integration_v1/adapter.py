"""Owned public sample copy, one row-index expression; no global Torch patch."""
import json,hashlib,inspect,textwrap,functools
from pathlib import Path
R=Path(__file__).resolve().parent;ATTR='_runner_hidden_gather_tileir_v1'
class Adapter:
 def __init__(self,output):self.output=Path(output);self.failed=False;self.mode=None;self.pending=False;self.rows=[];self.calls=0;self.current=None;self.shadow_ok=False
 def check(self):assert not self.failed,'Permanent first-failure latch'
 def start(self,mode):
  self.check();assert not self.pending and mode in ('shadow','replace')
  if mode=='replace':assert self.shadow_ok
  self.mode=mode;self.rows=[];self.calls=0
 def gather(self,x,idx):
  self.check();assert self.pending and self.current is None
  import torch
  from tileir_vllm_hidden_gather_v1 import gather_hidden_rows
  n=1696 if self.calls==1 else 32
  assert x.shape==(n,2560) and x.dtype==torch.bfloat16 and x.stride()==(2560,1) and x.storage_offset()==0 and x.untyped_storage().nbytes()==n*5120
  assert idx.shape==(32,) and idx.dtype==torch.int64 and idx.stride()==(1,) and idx.storage_offset()==0 and idx.untyped_storage().nbytes()==256
  def digest(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
  before=(digest(x),digest(idx));native=x[idx] if self.mode=='shadow' else None;tile=gather_hidden_rows(x,idx)
  def layout(y):return y.shape==(32,2560) and y.dtype==x.dtype and y.device==x.device and y.stride()==(2560,1) and y.storage_offset()==0 and y.untyped_storage().nbytes()==163840
  fresh=all(tile.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr()for t in (x,idx)) and (native is None or all(native.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr()for t in (x,idx,tile)))
  raw=None if native is None else torch.equal(tile.view(torch.int16),native.view(torch.int16));strict=None if native is None else bool(torch.allclose(tile.float(),native.float(),atol=1e-5,rtol=1e-5))
  finite=bool(torch.isfinite(tile).all()) and (native is None or bool(torch.isfinite(native).all()));unchanged=before==(digest(x),digest(idx))
  row=dict(call=self.calls,native_calls=int(native is not None),tile_calls=1,raw_equal=raw,strict=strict,finite=finite,layout=layout(tile)and(native is None or layout(native)),fresh=fresh,input_unchanged=unchanged)
  row['passed']=all(row[k]for k in ('finite','layout','fresh','input_unchanged'))and(native is None or(raw and strict));self.rows.append(row)
  if not row['passed']:
   self.failed=True;self.output.mkdir(parents=True,exist_ok=True);torch.save(dict(x=x.cpu(),indices=idx.cpu(),tile=tile.cpu(),native=None if native is None else native.cpu()),self.output/'first-failure.pt');raise AssertionError(row)
  self.after_check=lambda:before==(digest(x),digest(idx))
  self.current=native if native is not None else tile;return self.current
 def invoke(self,copied,runner,*a,**k):
  self.check();assert not self.pending and self.mode in ('shadow','replace');self.pending=True;self.current=None;self.calls+=1
  try:
   assert self.calls<=16 and runner.model is self.model and runner.batch_sharder is None
   ret=copied(runner,*a,**k);assert self.current is not None
   self.rows[-1]['post_method_inputs_preserved']=self.after_check();assert self.rows[-1]['post_method_inputs_preserved'];return ret
  except BaseException:self.failed=True;raise
  finally:self.pending=False
 def finish(self):
  self.check();assert not self.pending and self.calls==16 and len(self.rows)==16 and all(x['passed']for x in self.rows)
  if self.mode=='shadow':assert all(x['raw_equal'] and x['strict']for x in self.rows);self.shadow_ok=True
  return dict(shadow_pass=self.shadow_ok,supported_calls=16,native_gather_calls=sum(x['native_calls']for x in self.rows),tileir_gather_calls=16,supported_vendor_fallback_calls=0,rows=self.rows)
def install(model,output):
 import vllm.v1.worker.gpu.model_runner as mr
 assert not hasattr(model,ATTR);a=Adapter(output);a.model=model;old=mr.GPUModelRunner.sample
 pin=json.loads((R/'SOURCE.json').read_text());assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
 assert textwrap.dedent(inspect.getsource(old)).strip()==(R/'sample.original.py.txt').read_text().strip()
 ns=dict(old.__globals__);ns['_gather']=a.gather;exec(compile('from __future__ import annotations\n'+(R/'sample.copied.py.txt').read_text(),str(R/'sample.copied.py.txt'),'exec'),ns);copied=ns['sample']
 @functools.wraps(old)
 def wrapped(runner,*args,**kwargs):return a.invoke(copied,runner,*args,**kwargs)
 a.owner=mr.GPUModelRunner;a.original=old;a.wrapper=wrapped;mr.GPUModelRunner.sample=wrapped;setattr(model,ATTR,a);return dict(installed=True)
def start(model,mode):getattr(model,ATTR).start(mode);return dict(started=True)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 assert a.owner.sample is a.wrapper;a.owner.sample=a.original;delattr(model,ATTR);return dict(restored=True)
