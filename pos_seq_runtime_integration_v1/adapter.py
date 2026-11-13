"""Public runner/helper adapter; original flags and unrelated operations unchanged."""
from pathlib import Path
from collections import Counter
import json,hashlib
from pos_seq_runtime_capture_v2.observer import state_metadata
from norm_runtime_qualification_v2.run import identity,aliases,descriptor
from tileir_vllm_pos_seq_v1.api import reference_checked
NAMES=('idx_mapping','query_start_loc','num_computed_tokens','pos','seq_lens')
ATTR='_tileir_pos_seq_integration_v1'
class Adapter:
 def __init__(self,out):
  self.out=Path(out);self.out.mkdir(exist_ok=False);self.bindings=[];self.pending=None;self.mode='shadow';self.rows=[];self.shadow_pass=False;self.native_calls=0;self.tileir_calls=0;self.current_call=None
 def save(self):(self.out/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2)+'\n')
 def install_classes(self,runner_cls,module):
  original_prepare=runner_cls.prepare_inputs;original=module.prepare_pos_seq_lens
  manifest=Path(__file__).resolve().parent.parent/'pos_seq_runtime_capture_v2/posSeqCapture02/bank/MANIFEST.json';source=json.loads(manifest.read_text())['samples'][0]['source']
  assert identity(original)==source['helper'] and identity(original_prepare)==source['prepare']
  def prepare(runner,scheduler_output,batch_req_state,batch_desc,num_active_loras):
   assert self.pending is None and runner.pcp_manager is None
   ctx=dict(runner=runner,batch=state_metadata(batch_req_state),calls=0);self.pending=ctx
   try:
    ret=original_prepare(runner,scheduler_output,batch_req_state,batch_desc,num_active_loras);assert ctx['calls']==1;return ret
   finally:self.pending=None
  def helper(idx_mapping,query_start_loc,num_computed_tokens,pos,seq_lens):
   assert self.pending is not None,'Unexpected unassociated helper; no fallback'
   ctx=self.pending;ctx['calls']+=1;assert ctx['calls']==1;runner=ctx['runner'];phase=ctx['batch']['phase'];assert phase in ('prefill','decode')
   assert pos is runner.input_buffers.positions and seq_lens is runner.input_buffers.seq_lens and num_computed_tokens is runner.req_states.num_computed_tokens.gpu
   assert not hasattr(runner.req_states.num_computed_tokens,'_uva_buf'),'Observed regular device scope only'
   import torch
   from tileir_vllm_pos_seq_v1 import prepare_pos_seq_lens as tile
   vals=dict(zip(NAMES,(idx_mapping,query_start_loc,num_computed_tokens,pos,seq_lens)))
   assert not any(v['storage_overlap'] for v in aliases(vals).values())
   args=[t.detach().cpu().tolist() for t in vals.values()];expected_values=reference_checked(*args);assert args[0]==ctx['batch']['idx_mapping']
   self.current_call=dict(mode=self.mode,phase=phase,index=len(self.rows))
   def raw(t):return torch.empty(0,dtype=torch.uint8,device=t.device).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,))
   before={k:raw(t).clone() for k,t in vals.items()};descs={k:descriptor(t) for k,t in vals.items()};prealias=aliases(vals)
   if self.mode=='shadow':
    clones={k:torch.empty(0,dtype=t.dtype,device=t.device).set_(before[k].clone().untyped_storage(),t.storage_offset(),t.shape,t.stride()) for k,t in vals.items()}
    native_return=original(*(vals[k] for k in NAMES));self.native_calls+=1
    candidate_return=tile(*(clones[k] for k in NAMES));self.tileir_calls+=1
    torch.cuda.synchronize()
    full={k:bool(torch.equal(raw(vals[k]),raw(clones[k]))) for k in NAMES}
    inputs={k:bool(torch.equal(raw(vals[k]),before[k])) and bool(torch.equal(raw(clones[k]),before[k])) for k in NAMES[:3]}
    ok=native_return is None and candidate_return is None and all(full.values()) and all(inputs.values()) and aliases(clones)==prealias==aliases(vals) and descs=={k:descriptor(t) for k,t in vals.items()}
    row=dict(phase=phase,mode='shadow',full_storage_equal=full,input_storage_preserved=inputs,passed=ok,metadata=descs)
   else:
    assert self.shadow_pass
    ret=tile(*(vals[k] for k in NAMES));self.tileir_calls+=1;torch.cuda.synchronize()
    inputs={k:bool(torch.equal(raw(vals[k]),before[k])) for k in NAMES[:3]}
    expected_bytes={}
    for k,items in zip(('pos','seq_lens'),expected_values):
     buf=before[k].clone();t=vals[k];view=torch.empty(0,dtype=t.dtype,device=t.device).set_(buf.untyped_storage(),t.storage_offset(),t.shape,t.stride());view.copy_(torch.tensor(items,dtype=t.dtype,device=t.device));expected_bytes[k]=bool(torch.equal(raw(t),buf))
    ok=ret is None and all(expected_bytes.values()) and all(inputs.values()) and aliases(vals)==prealias and descs=={k:descriptor(t) for k,t in vals.items()}
    row=dict(phase=phase,mode='replace',passed=ok,output_full_storage_cpu_oracle=expected_bytes,input_storage_preserved=inputs,metadata=descs)
   self.rows.append(row);self.current_call=None;self.save();assert ok,'Position/sequence contract mismatch';return None
  for owner,name,old,new in [(runner_cls,'prepare_inputs',original_prepare,prepare),(module,'prepare_pos_seq_lens',original,helper)]:self.bindings.append((owner,name,old,new));setattr(owner,name,new)
  return dict(source=source,scope='Public helper replacement, regular CUDA, no unrelated flags changed')
 def start(self,mode):
  assert mode in ('shadow','replace') and self.pending is None
  if mode=='replace':assert self.shadow_pass
  self.mode=mode;self.rows=[];self.native_calls=0;self.tileir_calls=0;return dict(mode=mode)
 def summary(self):return dict(mode=self.mode,rows=self.rows,supported_calls=len(self.rows),native_calls=self.native_calls,tileir_calls=self.tileir_calls,shadow_pass=self.shadow_pass,current_call=self.current_call)
 def finish(self):
  assert Counter(x['phase'] for x in self.rows)==Counter(prefill=1,decode=15) and len(self.rows)==16
  assert self.tileir_calls==16 and self.native_calls==(16 if self.mode=='shadow' else 0)
  if self.mode=='shadow':self.shadow_pass=all(x['passed'] for x in self.rows)
  self.save();return self.summary()
 def close(self):
  conflicts=[]
  for owner,name,old,new in reversed(self.bindings):
   if getattr(owner,name) is new:setattr(owner,name,old)
   else:conflicts.append(name)
  self.bindings=[];assert not conflicts;return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR)
 import vllm.v1.worker.gpu.model_runner as module
 a=Adapter(output)
 try:r=a.install_classes(module.GPUModelRunner,module);setattr(model,ATTR,a);return r
 except BaseException:a.close();raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 r=a.close();delattr(model,ATTR);return r
