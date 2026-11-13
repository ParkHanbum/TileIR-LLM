"""Scoped CPU-controlled public prepare_attn/gather boundary observation."""
from pathlib import Path
import hashlib,json
ATTR='_tileir_block_gather_observer_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metadata(t):
 s=t.untyped_storage()
 return dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),dtype=str(t.dtype),device=str(t.device),data_ptr=t.data_ptr(),storage_ptr=s.data_ptr(),storage_bytes=s.nbytes(),is_pinned=t.is_pinned(),requires_grad=t.requires_grad)
def phase_metadata(b):
 flags=[bool(x) for x in b.is_prefilling_np.tolist()]
 assert len(flags)==b.num_reqs and bool(b.has_prefill)==any(flags)
 phase='prefill' if flags and all(flags) else 'decode' if flags and not any(flags) else 'mixed'
 return dict(phase=phase,num_reqs=int(b.num_reqs),num_reqs_after_padding=int(b.num_reqs_after_padding),num_tokens=int(b.num_tokens),num_tokens_after_padding=int(b.num_tokens_after_padding),req_ids=list(b.req_ids),has_prefill=bool(b.has_prefill),is_prefilling=flags,num_scheduled_tokens=b.num_scheduled_tokens.tolist(),num_computed_tokens=b.num_computed_tokens_np.tolist(),prefill_len=b.prefill_len_np.tolist(),num_computed_prefill_tokens=b.num_computed_prefill_tokens_np.tolist(),query_start_loc=b.query_start_loc_np.tolist(),idx_mapping=b.idx_mapping_np.tolist())
class Observer:
 def __init__(self,out):
  self.out=Path(out);self.out.mkdir(parents=True,exist_ok=False)
  self.samples={};self.seen={};self.pending=None;self.bindings=[];self.errors=[];self.counter=0;self.closed=False
 def tensor(self,t,label):
  import torch
  meta=metadata(t);logical=t.detach().cpu().clone()
  logicalpath=self.out/(label+'.pt');torch.save(logical,logicalpath)
  # Capture the ENTIRE byte allocation, including nonlogical tails/offsets.
  raw=torch.empty(0,dtype=torch.uint8,device=t.device).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,)).cpu().clone()
  rawpath=self.out/(label+'.storage.pt');torch.save(raw,rawpath)
  return dict(metadata=meta,path=str(logicalpath),sha256=sha(logicalpath),logical_sha256=hashlib.sha256(logical.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),storage_path=str(rawpath),storage_sha256=sha(rawpath),storage_byte_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest())
 def snapshot(self,owner,idx,label):
  values={'indices':idx,'counts_device_view':owner.num_blocks.gpu,'counts_host_source':owner.num_blocks.cpu}
  values.update({'src_'+str(g):t.gpu for g,t in enumerate(owner.block_tables)})
  values.update({'out_'+str(g):t for g,t in enumerate(owner.input_block_tables)})
  pool=owner.num_blocks.pool;active=pool._curr;buf=pool._uva_bufs[active]
  values['counts_active_host']=buf.cpu;values['counts_active_accelerator_view']=buf._uva
  placement=dict(pool_class=type(pool).__name__,active_index=active,buffer_class=type(buf).__name__,cpu=metadata(buf.cpu),accelerator_view=metadata(buf._uva),counts_view=metadata(owner.num_blocks.gpu),device_view_shares_active_storage=owner.num_blocks.gpu.untyped_storage().data_ptr()==buf._uva.untyped_storage().data_ptr(),note='UvaBuffer denotes mapped pinned host accelerator view; NonUvaBuffer denotes explicit copied device fallback; device label alone is not physical placement')
  return {k:self.tensor(v,label+'_'+k) for k,v in values.items()},placement
 def progress(self):
  r=dict(schema='block-gather-capture-v1',complete=set(self.samples)=={'prefill','decode'} and not self.pending and not self.errors,samples=list(self.samples.values()),observed_calls=self.seen,errors=self.errors,observers_removed=self.closed,scope='Eager native public calls; class boundary observation, no replacement/graph/optimized fusion claim')
  (self.out/'PROGRESS.json').write_text(json.dumps(r,indent=2));return r
 def install_classes(self,runner_cls,block_cls):
  assert not self.bindings
  original_prepare=runner_cls.prepare_attn;original_gather=block_cls.gather_block_tables
  def prepare(runner,batch,*args,**kwargs):
   assert self.pending is None,'Nested prepare_attn is outside scope'
   assert runner.pcp_manager is None,'PCP boundary unsupported'
   context=dict(owner=runner.block_tables,batch=batch,metadata=phase_metadata(batch),gathers=0)
   self.pending=context
   try:
    ret=original_prepare(runner,batch,*args,**kwargs)
    assert context['gathers']==1,'Expected exactly one matching gather per prepare_attn'
    return ret
   finally:self.pending=None
  def gather(owner,idx,num_reqs_padded,out=None,out_ptrs=None):
   ctx=self.pending
   if ctx is None or owner is not ctx['owner']:return original_gather(owner,idx,num_reqs_padded,out=out,out_ptrs=out_ptrs)
   ctx['gathers']+=1
   assert idx is ctx['batch'].idx_mapping and num_reqs_padded==ctx['metadata']['num_reqs_after_padding']
   assert out is None and out_ptrs is None,'Only default persistent buffers supported'
   phase=ctx['metadata']['phase'];self.seen[phase]=self.seen.get(phase,0)+1
   if phase not in ('prefill','decode') or phase in self.samples:
    return original_gather(owner,idx,num_reqs_padded,out=out,out_ptrs=out_ptrs)
   self.counter+=1;label=f'{self.counter:02d}_{phase}'
   try:
    pre,placement=self.snapshot(owner,idx,label+'_pre');self.progress()
    ret=original_gather(owner,idx,num_reqs_padded,out=out,out_ptrs=out_ptrs)
    post,placement_after=self.snapshot(owner,idx,label+'_post')
    assert isinstance(ret,tuple) and len(ret)==len(owner.input_block_tables)
    outputs=[self.tensor(t,label+'_return_'+str(g)) for g,t in enumerate(ret)]
    preserved={k:pre[k]['storage_byte_sha256']==post[k]['storage_byte_sha256'] for k in pre if not k.startswith('out_')}
    assert all(preserved.values()),'Native gather mutated input backing storage'
    for g,t in enumerate(ret):
     assert t.untyped_storage().data_ptr()==owner.input_block_tables[g].untyped_storage().data_ptr()
     assert tuple(t.shape)==(num_reqs_padded,owner.input_block_tables[g].shape[1])
    self.samples[phase]=dict(phase=phase,batch=ctx['metadata'],association=['GPUModelRunner.prepare_attn','same_runner.block_tables.gather_block_tables'],original_call_count=1,pre=pre,post=post,outputs=outputs,input_storage_preserved=preserved,counts_placement=placement,counts_placement_after=placement_after,return_preserved=True)
    self.progress();return ret
   except BaseException as e:self.errors.append(repr(e));self.progress();raise
  for cls,name,old,new in ((runner_cls,'prepare_attn',original_prepare,prepare),(block_cls,'gather_block_tables',original_gather,gather)):
   self.bindings.append((cls,name,old,new));setattr(cls,name,new)
 def close(self):
  conflicts=[]
  for cls,name,old,new in reversed(self.bindings):
   if getattr(cls,name) is new:setattr(cls,name,old)
   else:conflicts.append(name)
  self.bindings=[];self.closed=not conflicts
  if conflicts:raise RuntimeError('Foreign bindings not overwritten: '+str(conflicts))
 def finish(self):
  self.close();r=self.progress()
  r['source_pins']=json.loads(Path(__file__).with_name('PINS.json').read_text())
  with (self.out/'MANIFEST.json').open('x') as f:json.dump(r,f,indent=2)
  r['manifest_path']=str(self.out/'MANIFEST.json');r['manifest_sha256']=sha(self.out/'MANIFEST.json')
  return r
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
