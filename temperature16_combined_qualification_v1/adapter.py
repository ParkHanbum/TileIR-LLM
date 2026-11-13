from pathlib import Path
import hashlib,inspect,textwrap
from lean_sampling144_v1 import adapter as lean
from sampling_guards144_runtime_v2 import adapter as proof
ATTR=lean.ATTR
R=Path(__file__).resolve().parent
class TempChecks:
 def __init__(self,c):self.controller=c;self.active=False;self.shadow_pass=False;self.busy=False
 def start(self,mode):
  assert not self.active and self.controller.failed is None and mode in('shadow','replace')and(mode=='shadow'or self.shadow_pass)
  self.mode=mode;self.rows=[];self.active=True
 def predicate(self,temp,idx):
  import torch
  from public_temperature_zero_v1.api import selected_all_zero,selected_1d_checked
  row=None
  try:
   assert self.active and self.controller.active and self.controller.failed is None and not self.busy and len(self.rows)<16
   for x,dtype,sz in ((temp,torch.float32,128),(idx,torch.int64,256)):
    assert tuple(x.shape)==(32,)and tuple(x.stride())==(1,)and x.storage_offset()==0 and x.untyped_storage().nbytes()==sz
    assert x.dtype==dtype and x.device.type=='cuda'and not x.requires_grad and not x.is_neg()and not x.is_conj()
   assert temp.device==idx.device
   self.busy=True;row=dict(index=len(self.rows)+1,mode=self.mode,stage='entered');self.rows.append(row)
   a=temp.view(torch.uint8).cpu();b=idx.view(torch.uint8).cpu();stream=torch.cuda.current_stream(temp.device).cuda_stream
   native=None
   if self.mode=='shadow':
    selected=temp.index_select(0,idx);native=bool((selected==0).all())
    # Direct raw gather output proof before subsequent equality conversion.
    tile_selected=selected_1d_checked(temp,idx)
    assert torch.equal(selected.view(torch.uint8),tile_selected.view(torch.uint8))
   value=selected_all_zero(temp,idx);assert type(value)is bool and(native is None or value is native)
   assert torch.equal(a,temp.view(torch.uint8).cpu())and torch.equal(b,idx.view(torch.uint8).cpu())and torch.cuda.current_stream(temp.device).cuda_stream==stream
   row.update(stage='verified',native=native,tile=value,readonly=True,same_stream=True,raw_gather_compared=self.mode=='shadow')
   return native if self.mode=='shadow'else value
  except BaseException as e:
   if row is not None:row['error']=repr(e)
   self.controller.latch('temperature16',e);raise
  finally:self.busy=False
 def finish(self):
  assert self.active and not self.busy and self.controller.failed is None and len(self.rows)==16 and all(r['stage']=='verified'for r in self.rows)
  self.active=False
  if self.mode=='shadow':self.shadow_pass=True
  return dict(calls=16,mode=self.mode,rows=self.rows,shadow_pass=self.shadow_pass,native_predicates=16 if self.mode=='shadow'else 0,extra_checked_range_calls=32 if self.mode=='shadow'else 16)
class Greedy(proof.Greedy):
 def install(self,model):
  result=super().install(model)
  old=self.range_body;source_path=Path(proof.__file__).with_name('range_greedy.candidate.py.txt')
  assert old.__code__.co_filename==str(source_path)and old.__code__.co_firstlineno==1 and not old.__code__.co_freevars
  source=source_path.read_text();assert textwrap.dedent(inspect.getsource(old))==source
  assert hashlib.sha256(source.encode()).hexdigest()=='c2a51f203ef5646f0cdfdabae70d719f168de5257f2e5de58ed5c56b8622b4c3'
  expression='bool((temp.index_select(0,idx)==0).all())';assert source.count(expression)==1
  candidate=source.replace(expression,'_temperature_zero(temp,idx)');assert candidate==(R/'temperature_owned.py.txt').read_text()
  ns=dict(old.__globals__);ns['_temperature_zero']=self.controller.temperature16.predicate
  exec(compile(candidate,str(R/'temperature_owned.py.txt'),'exec'),ns)
  self.range_body=ns['apply'];assert self.range_body.__globals__['_range_guard']is old.__globals__['_range_guard']
  result['temperature_copy_sha256']=hashlib.sha256(candidate.encode()).hexdigest();return result
SPECS=tuple((key,Greedy if key=='greedy_sample'else cls,n)for key,cls,n in proof.SPECS)
class Controller(lean.Controller):
 def __init__(self,model):
  proof.ranged.base.Controller.__init__(self,model,SPECS)
  self.range48=lean.Range(self);self.sampling96=lean.Sampling(self);self.temperature16=TempChecks(self)
 def start_mode(self,mode):
  self.temperature16.start(mode);return super().start_mode('replace')
 def finish(self):
  t=self.temperature16.finish();d=super().finish();d['temperature16']=t;d['sampling96']['temperature_original_retained']=16 if t['mode']=='shadow'else 0;d['sampling96']['temperature_followup_hook']=True;return d
def install(model):
 assert not hasattr(model,ATTR);c=Controller(model);setattr(model,ATTR,c);return c.install()
def start(model,mode):return getattr(model,ATTR).start_mode(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):return lean.close(model)
def release(model):return lean.release(model)
