from pathlib import Path
import json,functools,hashlib,struct
from residual_zero_prefill_public_v1.contract import prefill_reference
from tileir_prepare_prefill_inputs_v1.api import prepare_prefill_inputs,validate
ATTR='_prepare_prefill_inputs_runtime_replace_v1'
SOURCE_SHA='2e4c2ebdb2c1ce04026d15288f5408624d880f335735636985efa3a3209a56f2'
def raw(t):
 import torch
 return t.detach().view(torch.uint8).cpu().numpy().tobytes()
class Adapter:
 def __init__(self,output):
  self.out=Path(output);self.out.mkdir(exist_ok=False);self.mode=None;self.rows=[];self.failed=None;self.baseline=False;self.binding=None
 def guard(self):assert self.failed is None,('Latched',self.failed)
 def save(self):
  d=dict(mode=self.mode,rows=self.rows,failed=self.failed,shadow_only=False,replacement=self.mode=="replace")
  (self.out/((self.mode or 'install')+'-PROGRESS.json')).write_text(json.dumps(d,indent=2));return d
 def fail(self,e):
  if self.failed is None:self.failed=repr(e)
  self.save()
 def invoke(self,native,args):
  self.guard()
  try:
   assert len(self.rows)==0 and self.mode in ('shadow','replace')
   validate(args);before=[raw(t)for t in args];row=dict(passed=False,native_calls=0,tile_calls=0);self.rows.append(row);self.save()
   def ints(i,code='i'):return list(struct.unpack('<'+code*(len(before[i])//struct.calcsize(code)),before[i]))
   expected_input=ints(0);flat=ints(4);all_ids=[flat[i*128:(i+1)*128]for i in range(32)];expected_next=[ints(1)]
   prefill_reference(expected_input,expected_next,ints(2,'q'),ints(3),all_ids,ints(5),ints(6))
   expected=[struct.pack('<'+'i'*len(expected_input),*expected_input),struct.pack('<'+'i'*32,*expected_next[0])]
   signatures=[(id(t),t.data_ptr(),t.untyped_storage().data_ptr(),t.storage_offset()) for t in args]
   if self.mode=='shadow':ret=native(*args);row['native_calls']=1
   else:ret=prepare_prefill_inputs(*args);row['tile_calls']=1
   assert ret is None
   assert signatures==[(id(t),t.data_ptr(),t.untyped_storage().data_ptr(),t.storage_offset())for t in args],'Original buffers replaced'
   assert all(raw(args[i])==expected[i]for i in (0,1)),'Full output CPU-source oracle mismatch'
   assert all(raw(args[i])==before[i]for i in range(2,7)),'Readonly source modified'
   row.update(passed=True,outputs_cpu_reference_raw=True,readonly_raw=True,original_output_addresses=True,output_bytes=len(before[0])+len(before[1]),readonly_bytes=sum(map(len,before[2:])),native_oracle_calls=0)
   self.save();return ret
  except BaseException as e:self.fail(e);raise
 def install(self,model):
  import vllm.v1.worker.gpu.model_runner as m
  import vllm.v1.worker.gpu.input_batch as src
  assert hashlib.sha256(Path(src.__file__).read_bytes()).hexdigest()==SOURCE_SHA
  old=m.prepare_prefill_inputs;assert old is src.prepare_prefill_inputs
  @functools.wraps(old)
  def wrapped(*args,**kwargs):
   self.guard()
   try:
    assert not kwargs
    return self.invoke(old,args)
   except BaseException as e:self.fail(e);raise
  self.binding=(m,old,wrapped);m.prepare_prefill_inputs=wrapped;return dict(shadow_only=False,original_identity=True,native_oracle_calls=0)
 def start(self,mode):
  self.guard()
  try:assert mode in ('shadow','replace') and (mode!='replace' or self.baseline);self.mode=mode;self.rows=[];return dict(shadow_only=False)
  except BaseException as e:self.fail(e);raise
 def finish(self):
  self.guard()
  try:
   from .result_gate import native_finish_gate
   d=self.save();assert native_finish_gate(d)
   if self.mode=='shadow':self.baseline=True
   return d
  except BaseException as e:self.fail(e);raise
 def close(self):
  if self.binding:
   m,old,new=self.binding;assert m.prepare_prefill_inputs is new;m.prepare_prefill_inputs=old;self.binding=None
  return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR);a=Adapter(output);setattr(model,ATTR,a)
 try:return a.install(model)
 except BaseException:a.close();delattr(model,ATTR);raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 try:return a.close()
 finally:delattr(model,ATTR)
