from lean_public_composition_v1.support import Base
from tileir_prepare_prefill_inputs_v1 import prepare_prefill_inputs
import inspect,hashlib
from pathlib import Path
SOURCE_SHA='2e4c2ebdb2c1ce04026d15288f5408624d880f335735636985efa3a3209a56f2'
def check_owned_bridge(part,wrapped):
 bindings=[b for b in part.bindings if b[1]=='prepare_inputs'];assert len(bindings)==1
 guarded=bindings[0][5];new=guarded.__wrapped__;copied=inspect.getclosurevars(new).nonlocals.get('copied')
 assert copied is not None and copied.__globals__['prepare_prefill_inputs'] is wrapped,'Owned prepare_inputs captured different prefill helper'
 return True
class PrefillInputs(Base):
 def install(self,model):
  import vllm.v1.worker.gpu.model_runner as mr
  import vllm.v1.worker.gpu.input_batch as src
  assert hashlib.sha256(Path(src.__file__).read_bytes()).hexdigest()==SOURCE_SHA
  assert mr.prepare_prefill_inputs is src.prepare_prefill_inputs
  self.bind(mr,'prepare_prefill_inputs',self.invoke);self.wrapped=self.bindings[-1][5]
  return dict(boundary='prepare_prefill_inputs',binding_before_owned_prepare_inputs=True,numerical_api='tileir_prepare_prefill_inputs_v1')
 def invoke(self,*args,**kwargs):
  self.guard();assert not kwargs and len(args)==7
  # Keep original readonly UVA owner tensors reachable through caller-synchronized release.
  self.controller.owner_refs.extend(args)
  signatures=[(id(t),t.data_ptr(),t.untyped_storage().data_ptr(),t.storage_offset()) for t in args]
  ret=self.call('runner','prefill',prepare_prefill_inputs,*args)
  assert ret is None and signatures==[(id(t),t.data_ptr(),t.untyped_storage().data_ptr(),t.storage_offset()) for t in args]
  return ret
 def finish(self):
  assert dict(self.counts)=={('runner','prefill'):1}
  return super().finish()
