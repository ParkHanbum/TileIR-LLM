"""Public prepare_inputs/helper association; no replacement or inferred phases."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from block_gather_runtime_capture_v1.observer import Observer as Base,metadata,phase_metadata,sha
from norm_runtime_qualification_v2.run import identity,aliases
ATTR='_tileir_pos_seq_capture_v1'
def state_metadata(b):
 flags=[bool(x) for x in b.is_prefilling_np.tolist()];assert len(flags)==len(b.req_ids) and any(flags)==bool(b.has_prefill)
 return dict(phase='prefill' if flags and all(flags) else 'decode' if flags and not any(flags) else 'mixed',req_ids=list(b.req_ids),is_prefilling=flags,has_prefill=bool(b.has_prefill),num_tokens=int(b.num_tokens),idx_mapping=b.idx_mapping_np.tolist(),num_scheduled_tokens=b.num_scheduled_tokens.tolist(),prefill_len=b.prefill_len_np.tolist(),num_computed_prefill_tokens=b.num_computed_prefill_tokens_np.tolist())
class Observer(Base):
 def progress(self):
  r=dict(schema='pos-seq-capture-v1',complete=set(self.samples)=={'prefill','decode'} and not self.pending and not self.errors,samples=list(self.samples.values()),observed_calls=self.seen,errors=self.errors,observers_removed=self.closed,scope='Public eager prepare_inputs associated helper; full5 tensor backing storages before/after; no replacement')
  (self.out/'PROGRESS.json').write_text(json.dumps(r,indent=2));return r
 def snapshot(self,values,label):return {k:self.tensor(t,label+'_'+k) for k,t in values.items()}
 def install_classes(self,runner_cls,module):
  assert not self.bindings
  original_prepare=runner_cls.prepare_inputs;original_helper=module.prepare_pos_seq_lens
  self.identities=dict(prepare=identity(original_prepare),helper=identity(original_helper))
  def prepare(runner,scheduler_output,batch_req_state,batch_desc,num_active_loras):
   assert self.pending is None and runner.pcp_manager is None
   ctx=dict(runner=runner,batch=state_metadata(batch_req_state),calls=0,sample=None);self.pending=ctx
   try:
    ret=original_prepare(runner,scheduler_output,batch_req_state,batch_desc,num_active_loras)
    assert ctx['calls']==1,'Expected single helper in prepare_inputs'
    actual=phase_metadata(ret);assert actual['phase']==ctx['batch']['phase'] and actual['idx_mapping']==ctx['batch']['idx_mapping'] and actual['req_ids']==ctx['batch']['req_ids']
    if ctx['sample'] is not None:
     s=ctx['sample'];s['returned_batch']=actual;self.samples[actual['phase']]=s
    return ret
   except BaseException as e:self.errors.append(repr(e));raise
   finally:self.pending=None;self.progress()
  def helper(idx_mapping,query_start_loc,num_computed_tokens,pos,seq_lens):
   ctx=self.pending
   if ctx is None:return original_helper(idx_mapping,query_start_loc,num_computed_tokens,pos,seq_lens)
   runner=ctx['runner'];ctx['calls']+=1;assert ctx['calls']==1
   assert pos is runner.input_buffers.positions and seq_lens is runner.input_buffers.seq_lens and num_computed_tokens is runner.req_states.num_computed_tokens.gpu
   phase=ctx['batch']['phase'];self.seen[phase]=self.seen.get(phase,0)+1
   if phase not in ('prefill','decode') or phase in self.samples:return original_helper(idx_mapping,query_start_loc,num_computed_tokens,pos,seq_lens)
   self.counter+=1;label=f'{self.counter:02d}_{phase}';values=dict(idx_mapping=idx_mapping,query_start_loc=query_start_loc,num_computed_tokens=num_computed_tokens,pos=pos,seq_lens=seq_lens)
   owner=runner.req_states.num_computed_tokens;uva=getattr(owner,'_uva_buf',None)
   placement=dict(owner_class=type(owner).__name__,has_uva_buffer=uva is not None,device_view=metadata(num_computed_tokens),note='Recorded owner construction distinguishes mapped host from regular device; device label alone is not physical placement')
   if uva is not None:placement.update(buffer_class=type(uva).__name__,host=metadata(uva.cpu),mapped=metadata(uva.uva()))
   before=self.snapshot(values,label+'_pre');prealiases=aliases(values)
   # CPU evidence from snapshot, not guessed row counts or order.
   import torch
   actual_mapping=torch.load(before['idx_mapping']['path'],weights_only=True).tolist()
   assert actual_mapping==ctx['batch']['idx_mapping']
   ret=original_helper(idx_mapping,query_start_loc,num_computed_tokens,pos,seq_lens)
   assert ret is None
   after=self.snapshot(values,label+'_post')
   preserved={k:before[k]['storage_byte_sha256']==after[k]['storage_byte_sha256'] for k in ('idx_mapping','query_start_loc','num_computed_tokens')};assert all(preserved.values())
   ctx['sample']=dict(phase=phase,batch=ctx['batch'],association=['GPUModelRunner.prepare_inputs','same invocation prepare_pos_seq_lens'],original_call_count=1,return_kind='None',return_preserved=True,source=self.identities,pre=before,post=after,pre_aliases=prealiases,post_aliases=aliases(values),input_storage_preserved=preserved,computed_placement=placement)
   return ret
  for owner,name,old,new in [(runner_cls,'prepare_inputs',original_prepare,prepare),(module,'prepare_pos_seq_lens',original_helper,helper)]:self.bindings.append((owner,name,old,new));setattr(owner,name,new)
def install(model,out):
 assert not hasattr(model,ATTR)
 import vllm.v1.worker.gpu.model_runner as module
 o=Observer(out);setattr(model,ATTR,o)
 try:o.install_classes(module.GPUModelRunner,module);return o.progress()
 except BaseException:o.close();delattr(model,ATTR);raise
def finish(model):
 o=getattr(model,ATTR,None)
 if o is None:return dict(observer_absent=True)
 try:
  o.close();r=o.progress();r['source_pins']=json.loads(Path(__file__).with_name('PINS.json').read_text());f=o.out/'MANIFEST.json'
  with f.open('x') as h:json.dump(r,h,indent=2)
  r['manifest_path']=str(f);r['manifest_sha256']=sha(f);return r
 finally:delattr(model,ATTR)
