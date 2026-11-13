"""Native public slot mapping observation; no substitution or extra native call."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from block_gather_runtime_capture_v1.observer import Observer as Base,phase_metadata,sha,metadata
ATTR='_tileir_slot_mapping_observer_v1'
class Observer(Base):
 def snapshot(self,owner,idx,starts,pos,label):
  vals=dict(indices=idx,starts=starts,positions=pos,sizes=owner.block_sizes_tensor,enabled=owner.slot_mapping_enabled,dcp_sharded=owner.dcp_sharded,table_ptrs=owner.block_table_ptrs,table_strides=owner.block_table_strides,out=owner.slot_mappings)
  vals.update({'src_'+str(g):t.gpu for g,t in enumerate(owner.block_tables)})
  return {k:self.tensor(t,label+'_'+k) for k,t in vals.items()}
 def progress(self):
  r=dict(schema='slot-mapping-capture-v1',complete=set(self.samples)=={'prefill','decode'} and not self.pending and not self.errors,samples=list(self.samples.values()),observed_calls=self.seen,errors=self.errors,observers_removed=self.closed,scope='Eager native public compute_slot_mappings, first actual prefill/decode; no optimized graph equivalence')
  (self.out/'PROGRESS.json').write_text(json.dumps(r,indent=2));return r
 def install_classes(self,R,B):
  rp=R.prepare_attn;sm=B.compute_slot_mappings
  def prepare(r,b,*args,**kwargs):
   assert self.pending is None and r.pcp_manager is None
   ctx=dict(owner=r.block_tables,batch=b,metadata=phase_metadata(b),calls=0);self.pending=ctx
   try:
    ret=rp(r,b,*args,**kwargs);assert ctx['calls']==1;return ret
   finally:self.pending=None
  def slots(owner,idx_mapping,query_start_loc,positions,num_tokens_padded,out=None):
   ctx=self.pending
   if ctx is None or owner is not ctx['owner']:return sm(owner,idx_mapping,query_start_loc,positions,num_tokens_padded,out=out)
   ctx['calls']+=1;b=ctx['batch']
   assert idx_mapping is b.idx_mapping and query_start_loc is b.query_start_loc and positions is b.positions and num_tokens_padded==b.num_tokens_after_padding and out is None
   phase=ctx['metadata']['phase'];self.seen[phase]=self.seen.get(phase,0)+1
   if phase not in ('prefill','decode') or phase in self.samples:return sm(owner,idx_mapping,query_start_loc,positions,num_tokens_padded,out=out)
   label=phase
   try:
    pre=self.snapshot(owner,idx_mapping,query_start_loc,positions,label+'_pre');self.progress()
    ret=sm(owner,idx_mapping,query_start_loc,positions,num_tokens_padded,out=out)
    post=self.snapshot(owner,idx_mapping,query_start_loc,positions,label+'_post')
    preserved={k:pre[k]['storage_byte_sha256']==post[k]['storage_byte_sha256'] for k in pre if k!='out'}
    assert all(preserved.values())
    assert tuple(ret.shape)==(owner.num_kv_cache_groups,num_tokens_padded) and ret.stride()==owner.slot_mappings.stride() and ret.storage_offset()==owner.slot_mappings.storage_offset() and ret.untyped_storage().data_ptr()==owner.slot_mappings.untyped_storage().data_ptr()
    self.samples[phase]=dict(phase=phase,batch=ctx['metadata'],cp_size=owner.cp_size,cp_rank=owner.cp_rank,cp_interleave=owner.cp_interleave,python_block_sizes=list(owner.block_sizes),python_enabled=list(owner._slot_mapping_enabled),association=['GPUModelRunner.prepare_attn','same_runner.block_tables.compute_slot_mappings'],original_call_count=1,pre=pre,post=post,output=self.tensor(ret,label+'_return'),input_storage_preserved=preserved,return_preserved=True)
    self.progress();return ret
   except BaseException as e:self.errors.append(repr(e));self.progress();raise
  for cls,name,old,new in ((R,'prepare_attn',rp,prepare),(B,'compute_slot_mappings',sm,slots)):
   self.bindings.append((cls,name,old,new));setattr(cls,name,new)
 def finish(self):
  self.close();r=self.progress();r['source_pins']=json.loads(Path(__file__).with_name('PINS.json').read_text())
  with (self.out/'MANIFEST.json').open('x') as f:json.dump(r,f,indent=2)
  r.update(manifest_path=str(self.out/'MANIFEST.json'),manifest_sha256=sha(self.out/'MANIFEST.json'));return r
def install(model,out):
 assert not hasattr(model,ATTR)
 from vllm.v1.worker.gpu.model_runner import GPUModelRunner
 from vllm.v1.worker.gpu.block_table import BlockTables
 o=Observer(out);setattr(model,ATTR,o)
 try:o.install_classes(GPUModelRunner,BlockTables);return o.progress()
 except BaseException:o.close();delattr(model,ATTR);raise
def finish(model):
 o=getattr(model,ATTR,None)
 if o is None:return dict(observer_absent=True)
 try:return o.finish()
 finally:delattr(model,ATTR)
