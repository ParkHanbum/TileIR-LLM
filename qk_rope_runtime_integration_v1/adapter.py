"""Public imported helper adapter. Original project method/forward are never copied."""
import inspect,json,re
from pathlib import Path
from collections import Counter
from norm_runtime_qualification_v2.run import logical,descriptor,aliases,identity
from norm_runtime_qualification_v2.compare import metrics
ATTR='_tileir_qk_rope_integration_v1'
SOURCE_SHA='09ea2a385259edc9feadc79e304e86a31582b9770f80b8003d2d641782405818'
HELPER_SHA='c201d88fd5e810a819cf671e8883037f53f5801211b8141e4801df1a4e5d558d'
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.bindings=[];self.pending=None;self.mode='shadow';self.shadow_pass=False;self.rows=[];self.native_gate_calls=0;self.tileir_gate_calls=0
 def install(self,model):
  import vllm.model_executor.models.qwen3_next as module
  cls=module.Qwen3NextAttention
  found=[(n,m) for n,m in model.named_modules() if n.endswith('self_attn') and type(m) is cls]
  assert len(found)==8 and {int(re.search(r'layers\.(\d+)',n)[1]) for n,m in found}==set(range(3,32,4))
  assert identity(cls._project_qkv_gate)['source_sha256']==SOURCE_SHA and identity(module.fused_qk_rmsnorm_rope_gate)['source_sha256']==HELPER_SHA
  self.paths=[n for n,m in found];self.install_bindings(found,cls,module)
  return dict(paths=self.paths,eligible_modules=8,scope='Original eager project and forward; imported helper only')
 def install_bindings(self,found,cls,module):
  original=cls._project_qkv_gate;fn=module.fused_qk_rmsnorm_rope_gate;sig=inspect.signature(fn);owners={id(m):n for n,m in found}
  def project(this,qkv,positions):
   if id(this) not in owners:return original(this,qkv,positions)
   assert self.pending is None and this.use_fused_qk_norm_rope_gate is True
   ctx=dict(path=owners[id(this)],owner=this,qkv=qkv,positions=positions,calls=0);self.pending=ctx
   try:
    ret=original(this,qkv,positions)
    assert ctx['calls']==1 and isinstance(ret,tuple) and len(ret)==4
    hr=ctx['helper_return'];assert ret[0] is hr[0] and ret[1] is hr[1] and ret[3] is hr[2]
    v=ret[2];assert v.untyped_storage().data_ptr()==qkv.untyped_storage().data_ptr()
    assert tuple(v.shape)==(qkv.shape[0],1024) and tuple(v.stride())==(10240,1) and v.storage_offset()==qkv.storage_offset()+9216
    ctx['row'].update(project_tuple_preserved=True,v_contract=True);self.rows.append(ctx['row']);self.save()
    return ret
   finally:self.pending=None
  def helper(*args,**kwargs):
   ctx=self.pending
   if ctx is None:return fn(*args,**kwargs)
   ctx['calls']+=1;assert ctx['calls']==1
   b=sig.bind(*args,**kwargs);b.apply_defaults();a=b.arguments
   assert a['positions'] is ctx['positions'] and a['q_weight'] is ctx['owner'].q_norm.weight and a['k_weight'] is ctx['owner'].k_norm.weight and a['cos_sin_cache'] is ctx['owner'].rotary_emb.cos_sin_cache
   out,row=self.apply(ctx,a,lambda:fn(*b.args,**b.kwargs));ctx['helper_return']=out;ctx['row']=row;return out
  for obj,name,old,new in ((cls,'_project_qkv_gate',original,project),(module,'fused_qk_rmsnorm_rope_gate',fn,helper)):
   self.bindings.append((obj,name,old,new));setattr(obj,name,new)
 def apply(self,ctx,a,native):
  import torch
  from tileir_vllm_qk_rope_v1 import fused_qk_rmsnorm_rope_gate
  vals={k:a[k] for k in ('q_gate','k','q_weight','k_weight','cos_sin_cache','positions')};vals['qkv']=ctx['qkv']
  assert vals['q_gate'].untyped_storage().data_ptr()==vals['k'].untyped_storage().data_ptr()==vals['qkv'].untyped_storage().data_ptr()
  assert tuple(vals['positions'].stride())==(2049,1)
  def storages():
   return {t.untyped_storage().data_ptr():logical(torch.empty(0,dtype=torch.uint8,device=t.device).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,)),torch) for t in vals.values()}
  before=storages();ref=None
  if self.mode=='shadow':ref=native();self.native_gate_calls+=1;assert storages()==before,'Native input mutation'
  else:assert self.mode=='replace' and self.shadow_pass
  args={k:(v.detach() if isinstance(v,torch.Tensor) else v) for k,v in a.items()}
  detach_contract={k:dict(original_requires_grad=bool(v.requires_grad),candidate_requires_grad=bool(args[k].requires_grad),same_storage=v.untyped_storage().data_ptr()==args[k].untyped_storage().data_ptr(),same_shape=tuple(v.shape)==tuple(args[k].shape),same_stride=tuple(v.stride())==tuple(args[k].stride()),same_offset=v.storage_offset()==args[k].storage_offset()) for k,v in a.items() if isinstance(v,torch.Tensor)}
  assert all(not d['candidate_requires_grad'] and all(d[n] for n in ('same_storage','same_shape','same_stride','same_offset')) for d in detach_contract.values())
  tile=fused_qk_rmsnorm_rope_gate(**args);self.tileir_gate_calls+=1
  assert isinstance(tile,tuple) and len(tile)==3
  aa=aliases(dict(vals,**dict(zip(('q_out','k_out','gate_out'),tile))))
  fresh=all(not v['storage_overlap'] for key,v in aa.items() if any(n in key.split('|') for n in ('q_out','k_out','gate_out')))
  m=ctx['qkv'].shape[0];layout=all(t.dtype==torch.bfloat16 and tuple(t.shape)==(m,n) and t.is_contiguous() and t.storage_offset()==0 for t,n in zip(tile,(4096,1024,4096)))
  gate=vals['q_gate'].reshape(m,16,512)[:,:,256:].reshape(m,4096)
  row=dict(path=ctx['path'],shape=[m,10240],phase='prefill' if m==1696 else 'decode',mode=self.mode,inputs_preserved=storages()==before,fresh_disjoint=fresh,layout_equal=layout,gate_copy_raw_equal=logical(tile[2],torch)==logical(gate,torch),input_descriptors={k:descriptor(t) for k,t in vals.items()},detach_contract=detach_contract,finite=all(bool(torch.isfinite(t).all()) for t in tile),comparisons={})
  row['passed']=all(row[k] for k in ('inputs_preserved','fresh_disjoint','layout_equal','gate_copy_raw_equal','finite'))
  if ref is not None:
   assert isinstance(ref,tuple) and len(ref)==3
   row['comparisons']={k:metrics(t,n) for k,t,n in zip(('q','k','gate'),tile,ref)}
   row['passed'] &= all(v['raw_equal'] and v['strict_pass'] and v['candidate_nonfinite']==v['reference_nonfinite']==0 for v in row['comparisons'].values())
  if not row['passed']:
   (self.output/'MISMATCH.json').write_text(json.dumps(row,indent=2)+'\n')
   for k,t in dict(vals,**dict(zip(('candidate_q','candidate_k','candidate_gate'),tile)),**({} if ref is None else dict(zip(('native_q','native_k','native_gate'),ref)))).items():torch.save(t.detach().cpu(),self.output/(k+'.pt'))
   raise AssertionError('QK raw/contract mismatch; replacement forbidden')
  return (ref if ref is not None else tile),row
 def start(self,mode):
  assert mode in ('shadow','replace') and self.pending is None
  if mode=='replace':assert self.shadow_pass
  self.mode=mode;self.rows=[];self.native_gate_calls=0;self.tileir_gate_calls=0;return dict(mode=mode)
 def summary(self):return dict(mode=self.mode,rows=self.rows,supported_calls=len(self.rows),native_gate_calls=self.native_gate_calls,tileir_gate_calls=self.tileir_gate_calls,shadow_pass=self.shadow_pass,supported_vendor_fallback_calls=0)
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2)+'\n')
 def finish(self):
  assert self.pending is None and Counter((x['path'],x['phase']) for x in self.rows)==Counter({(n,p):c for n in self.paths for p,c in [('prefill',1),('decode',15)]})
  assert self.tileir_gate_calls==128 and self.native_gate_calls==(128 if self.mode=='shadow' else 0)
  assert all(x['passed'] and x['v_contract'] and x['project_tuple_preserved'] for x in self.rows)
  if self.mode=='shadow':self.shadow_pass=True
  self.save();return self.summary()
 def close(self):
  conflicts=[]
  for obj,name,old,new in reversed(self.bindings):
   if getattr(obj,name) is not new:conflicts.append(name)
   else:setattr(obj,name,old)
  self.bindings=[];assert not conflicts,conflicts;return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR);a=Adapter(output)
 try:r=a.install(model);setattr(model,ATTR,a);return r
 except BaseException:a.close();raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 r=a.close();delattr(model,ATTR);return r
