"""Compose pos/seq pending context with the unchanged padding numeric adapter.
No closure mutation, source file modification, identity spoofing, or Torch patch.
"""
from pathlib import Path
import functools,hashlib,inspect,json,textwrap
from runner_padding_runtime_integration_v1 import adapter as padding
from pos_seq_runtime_integration_v1 import adapter as pos
ATTR=padding.ATTR

def combined_invoke(pending,adapter,copied,runner,scheduler_output,batch_req_state,batch_desc,num_active_loras):
 # Same context setup/one-call/finally contract as pos.Adapter.install_classes.prepare.
 assert pending.pending is None and runner.pcp_manager is None
 ctx=dict(runner=runner,batch=pos.state_metadata(batch_req_state),calls=0);pending.pending=ctx
 try:
  ret=adapter.invoke(copied,runner,scheduler_output,batch_req_state,batch_desc,num_active_loras)
  assert ctx['calls']==1;return ret
 finally:pending.pending=None

def install(model,output):
 import vllm.v1.worker.gpu.model_runner as mr
 assert not hasattr(model,ATTR);pending=getattr(model,pos.ATTR)
 matches=[r for r in pending.bindings if r[0] is mr.GPUModelRunner and r[1]=='prepare_inputs'];assert len(matches)==1
 owner,name,native,pos_wrapper=matches[0];current=owner.prepare_inputs
 assert getattr(current,'_composition_entry_key',None)=='pos_seq' and current.__wrapped__ is pos_wrapper
 spec=json.loads((padding.R/'SOURCE.json').read_text());assert hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()==spec['sha256']
 assert textwrap.dedent(inspect.getsource(native)).strip()==(padding.R/'prepare_inputs.original.py.txt').read_text().strip()
 a=padding.Adapter(output);a.original=native
 ns=dict(native.__globals__);ns['_construct']=a.construct;ns['_fill_padding']=a.fill_padding
 exec(compile('from __future__ import annotations\n'+(padding.R/'prepare_inputs.copied.py.txt').read_text(),str(padding.R/'prepare_inputs.copied.py.txt'),'exec'),ns);copied=ns['prepare_inputs']
 @functools.wraps(native)
 def wrapped(runner,scheduler_output,batch_req_state,batch_desc,num_active_loras):
  return combined_invoke(pending,a,copied,runner,scheduler_output,batch_req_state,batch_desc,num_active_loras)
 a.owner=owner;a.wrapper=wrapped;a.composition_prior=current;owner.prepare_inputs=wrapped;setattr(model,ATTR,a)
 return dict(installed=True,scope='Explicit pos/seq pending context bridge around unchanged padding copied method/API',numeric_adapter=str(padding.R/'adapter.py'))
def start(model,mode):return padding.start(model,mode)
def finish(model):return padding.finish(model)
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 assert a.owner.prepare_inputs is a.wrapper;a.owner.prepare_inputs=a.composition_prior;delattr(model,ATTR);return dict(restored=True)
