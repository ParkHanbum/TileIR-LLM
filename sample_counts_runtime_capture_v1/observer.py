"""Sampler __call__ -> bound public sample-count helper capture, no substitution."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from block_gather_runtime_capture_v1.observer import Observer as Base,phase_metadata,sha,metadata
ATTR='_tileir_sample_counts_observer_v1'
class Observer(Base):
 def snapshot(self,sampler,args,label):
  vals=dict(zip(('sampled','seq_lens','cu_logits','indices','prefill'),args))
  owner=sampler.req_states.prefill_len;pool=owner.pool;active=pool._uva_bufs[pool._curr]
  assert args[-1] is owner.gpu
  vals.update(prefill_host_source=owner.cpu,prefill_active_host=active.cpu,prefill_active_accelerator=active._uva)
  placement=dict(pool_class=type(pool).__name__,buffer_class=type(active).__name__,active_index=pool._curr,cpu=metadata(active.cpu),accelerator=metadata(active._uva),passed_view=metadata(args[-1]),same_active_backing=args[-1].untyped_storage().data_ptr()==active._uva.untyped_storage().data_ptr())
  return {k:self.tensor(t,label+'_'+k) for k,t in vals.items()},placement
 def progress(self):
  r=dict(schema='sample-count-capture-v1',complete=set(self.samples)=={'prefill','decode'} and not self.pending and not self.errors,samples=list(self.samples.values()),observed_calls=self.seen,errors=self.errors,observers_removed=self.closed,scope='Native eager Sampler __call__ bound helper; no sampling policy change')
  (self.out/'PROGRESS.json').write_text(json.dumps(r,indent=2));return r
 def install_classes(self,S,module):
  call=S.__call__;fn=module.get_num_sampled_and_rejected
  def sampler(this,logits,input_batch,*a,**kw):
   assert self.pending is None
   ctx=dict(owner=this,batch=input_batch,metadata=phase_metadata(input_batch),calls=0);self.pending=ctx
   try:
    result=call(this,logits,input_batch,*a,**kw);assert ctx['calls']==1;return result
   finally:self.pending=None
  def counts(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len):
   args=(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len);ctx=self.pending
   if ctx is None:return fn(*args)
   b=ctx['batch'];assert seq_lens is b.seq_lens and cu_num_logits is b.cu_num_logits and idx_mapping is b.idx_mapping
   ctx['calls']+=1;phase=ctx['metadata']['phase'];self.seen[phase]=self.seen.get(phase,0)+1
   if phase not in ('prefill','decode') or phase in self.samples:return fn(*args)
   try:
    pre,placement=self.snapshot(ctx['owner'],args,phase+'_pre');self.progress()
    ret=fn(*args)
    post,placement_after=self.snapshot(ctx['owner'],args,phase+'_post')
    assert isinstance(ret,tuple) and len(ret)==2 and ret[0] is num_sampled
    preserved={k:pre[k]['storage_byte_sha256']==post[k]['storage_byte_sha256'] for k in pre if k!='sampled'};assert all(preserved.values())
    s=ret[0].untyped_storage();r=ret[1].untyped_storage();assert not (s.data_ptr()<r.data_ptr()+r.nbytes() and r.data_ptr()<s.data_ptr()+s.nbytes())
    self.samples[phase]=dict(phase=phase,batch=ctx['metadata'],association=['Sampler.__call__','sampler module bound get_num_sampled_and_rejected'],original_call_count=1,pre=pre,post=post,outputs=[self.tensor(t,phase+'_return_'+str(i)) for i,t in enumerate(ret)],input_storage_preserved=preserved,return_preserved=True,first_return_same_object=True,rejected_disjoint=True,prefill_placement=placement,prefill_placement_after=placement_after)
    self.progress();return ret
   except BaseException as e:self.errors.append(repr(e));self.progress();raise
  for obj,name,old,new in ((S,'__call__',call,sampler),(module,'get_num_sampled_and_rejected',fn,counts)):
   self.bindings.append((obj,name,old,new));setattr(obj,name,new)
 def finish(self):
  self.close();r=self.progress();r['source_pins']=json.loads(Path(__file__).with_name('PINS.json').read_text())
  with (self.out/'MANIFEST.json').open('x') as f:json.dump(r,f,indent=2)
  r.update(manifest_path=str(self.out/'MANIFEST.json'),manifest_sha256=sha(self.out/'MANIFEST.json'));return r
def install(model,out):
 assert not hasattr(model,ATTR)
 import vllm.v1.worker.gpu.sample.sampler as module
 o=Observer(out);setattr(model,ATTR,o)
 try:o.install_classes(module.Sampler,module);return o.progress()
 except BaseException:o.close();delattr(model,ATTR);raise
def finish(model):
 o=getattr(model,ATTR,None)
 if o is None:return dict(observer_absent=True)
 try:return o.finish()
 finally:delattr(model,ATTR)
