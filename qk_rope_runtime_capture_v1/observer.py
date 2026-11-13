"""Public owner -> imported fused helper exactly-once capture. No implementation copy."""
from pathlib import Path
import hashlib,inspect,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from block_gather_runtime_capture_v1.observer import Observer as Base,metadata,sha
from norm_runtime_contract_capture_v1.observer import identity,aliases
ATTR='_tileir_qk_rope_capture_v1'
class Observer(Base):
 def __init__(self,out):
  super().__init__(out);self.modules={};self.persisted={};self.bytes_written=0
 def tensor(self,t,label):
  import torch
  m=metadata(t);m['stream']=int(torch.cuda.current_stream(t.device).cuda_stream)
  logical=t.detach().cpu().contiguous();raw=torch.empty(0,dtype=torch.uint8,device=t.device).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,)).cpu().clone()
  def save(v,kind):
   h=hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest();key=(kind,str(v.dtype),tuple(v.shape),h)
   if key not in self.persisted:
    self.bytes_written+=v.numel()*v.element_size();assert self.bytes_written<=512*1024*1024,'Capture logical+storage payload budget exceeded'
    f=self.out/(label+'.'+kind+'.pt');assert not f.exists();torch.save(v,f);self.persisted[key]=(str(f),sha(f))
   return (*self.persisted[key],h)
  p,ph,lh=save(logical,'logical');sp,sph,sh=save(raw,'storage')
  return dict(metadata=m,path=p,sha256=ph,logical_sha256=lh,storage_path=sp,storage_sha256=sph,storage_byte_sha256=sh)
 def progress(self):
  r=dict(schema='qk-rope-capture-v1',complete=set(self.samples)=={'prefill','decode'} and self.pending is None and not self.errors,samples=list(self.samples.values()),modules=self.modules,observed_calls=self.seen,errors=self.errors,observers_removed=self.closed,payload_bytes=self.bytes_written,scope='Native explicit fused helper in eager model; observed Boolean and owner association, no graph qualification')
  (self.out/'PROGRESS.json').write_text(json.dumps(r,indent=2)+'\n');return r
 def capture(self,ctx,fn,bound):
  import torch
  a=bound.arguments;phase=ctx['phase'];owner=ctx['owner']
  assert a['positions'] is ctx['positions'] and a['q_weight'] is owner.q_norm.weight and a['k_weight'] is owner.k_norm.weight and a['cos_sin_cache'] is owner.rotary_emb.cos_sin_cache
  vals=dict(qkv=ctx['qkv'],q_gate=a['q_gate'],k=a['k'],q_weight=a['q_weight'],k_weight=a['k_weight'],cache=a['cos_sin_cache'],positions=a['positions'])
  assert vals['q_gate'].untyped_storage().data_ptr()==vals['qkv'].untyped_storage().data_ptr()==vals['k'].untyped_storage().data_ptr()
  pre={k:self.tensor(v,phase+'_pre_'+k) for k,v in vals.items()};ret=fn(*bound.args,**bound.kwargs)
  assert isinstance(ret,tuple) and len(ret)==3
  post={k:self.tensor(v,phase+'_post_'+k) for k,v in vals.items()};preserved={k:pre[k]['storage_byte_sha256']==post[k]['storage_byte_sha256'] for k in vals};assert all(preserved.values())
  q,k,g=ret;expected_gate=vals['q_gate'].reshape(-1,a['num_q_heads'],2*a['head_dim'])[:,:,a['head_dim']:].reshape(g.shape)
  gate_exact=torch.equal(g.contiguous().view(torch.uint8),expected_gate.contiguous().view(torch.uint8));assert gate_exact
  aa=aliases(dict(vals,q_out=q,k_out=k,gate_out=g))
  for out in ('q_out','k_out','gate_out'):
   for inp in vals:
    pair='|'.join(sorted((out,inp)));assert not aa[pair]['storage_overlap']
  for left,right in [('q_out','k_out'),('q_out','gate_out'),('k_out','gate_out')]:assert not aa['|'.join(sorted((left,right)))]['storage_overlap']
  scalars={k:a[k] for k in ('eps','num_q_heads','num_kv_heads','head_dim','rotary_dim','mrope_section','norm_beta')}
  row=dict(phase=phase,phase_assignment='Fixed workload shape1696/32; not scheduler phase observation',module_path=ctx['path'],selected_fused=bool(owner.use_fused_qk_norm_rope_gate),association=['same module _project_qkv_gate','qwen3_next imported fused_qk_rmsnorm_rope_gate'],original_call_count=1,parameters=scalars,pre=pre,post=post,outputs=[self.tensor(v,phase+'_output_'+n) for n,v in zip(('q','k','gate'),ret)],aliases=aa,input_storage_preserved=preserved,gate_copy_raw_equal=bool(gate_exact),outputs_fresh_disjoint=True,return_preserved=True)
  ctx['captured_row']=row;ctx['helper_return']=ret
  return ret
 def install_bindings(self,modules,cls,module):
  original=cls._project_qkv_gate;fn=module.fused_qk_rmsnorm_rope_gate;sig=inspect.signature(fn);owners={id(m):(n,m) for n,m in modules};assert len(owners)==8
  for n,m in modules:self.modules[n]=dict(selected_fused=bool(m.use_fused_qk_norm_rope_gate),project=identity(original),helper=identity(fn),rotary_class=type(m.rotary_emb).__name__)
  def project(this,qkv,positions):
   if id(this) not in owners:return original(this,qkv,positions)
   assert self.pending is None,'Nested owner unsupported';n=owners[id(this)][0];M=int(qkv.shape[0]);assert M in (1696,32);phase='prefill' if M==1696 else 'decode'
   ctx=dict(owner=this,path=n,qkv=qkv,positions=positions,phase=phase,calls=0);self.pending=ctx
   try:
    ret=original(this,qkv,positions)
    assert this.use_fused_qk_norm_rope_gate is True and ctx['calls']==1,'Explicit fused branch not observed exactly once'
    if 'captured_row' in ctx:
     assert isinstance(ret,tuple) and len(ret)==4;hr=ctx['helper_return'];assert ret[0] is hr[0] and ret[1] is hr[1] and ret[3] is hr[2]
     v=ret[2];ctx['captured_row']['project_return_preserved']=True;ctx['captured_row']['v_descriptor']=metadata(v);ctx['captured_row']['v_aliases']=aliases(dict(qkv=qkv,v=v));assert ctx['captured_row']['v_aliases']['qkv|v']['same_storage']
     self.samples[phase]=ctx['captured_row']
    return ret
   except BaseException as e:self.errors.append(repr(e));raise
   finally:self.pending=None;self.progress()
  def helper(*args,**kwargs):
   ctx=self.pending
   if ctx is None:return fn(*args,**kwargs)
   ctx['calls']+=1;assert ctx['calls']==1
   key=ctx['path']+'|'+ctx['phase'];self.seen[key]=self.seen.get(key,0)+1
   if not ctx['path'].endswith('layers.3.self_attn') or ctx['phase'] in self.samples:return fn(*args,**kwargs)
   bound=sig.bind(*args,**kwargs);bound.apply_defaults();return self.capture(ctx,fn,bound)
  for obj,name,old,new in ((cls,'_project_qkv_gate',original,project),(module,'fused_qk_rmsnorm_rope_gate',fn,helper)):
   self.bindings.append((obj,name,old,new));setattr(obj,name,new)
 def finish(self):
  self.close();r=self.progress();r['source_pins']=json.loads(Path(__file__).with_name('PINS.json').read_text())
  with (self.out/'MANIFEST.json').open('x') as f:json.dump(r,f,indent=2)
  r.update(manifest_path=str(self.out/'MANIFEST.json'),manifest_sha256=sha(self.out/'MANIFEST.json'));return r
def install(model,out):
 assert not hasattr(model,ATTR)
 import vllm.model_executor.models.qwen3_next as module
 found=[(n,m) for n,m in model.named_modules() if n.endswith('self_attn') and type(m) is module.Qwen3NextAttention];assert len(found)==8
 o=Observer(out);setattr(model,ATTR,o)
 try:o.install_bindings(found,module.Qwen3NextAttention,module);return o.progress()
 except BaseException:o.close();delattr(model,ATTR);raise
def finish(model):
 o=getattr(model,ATTR,None)
 if o is None:return dict(observer_absent=True)
 try:return o.finish()
 finally:delattr(model,ATTR)
