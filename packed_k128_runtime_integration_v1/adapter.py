"""Only public instance ba Linear.apply bindings; all other GEMMs remain native."""
import hashlib,inspect,json,re,types
from collections import Counter
from pathlib import Path
ATTR='_tileir_ba_k128_runtime_v1'
LAYERS=tuple(i for i in range(32)if i%4!=3)

def source(fn):
 f=getattr(fn,'__func__',fn);p=inspect.getsourcefile(f)
 return dict(module=f.__module__,qualname=f.__qualname__,path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def metadata(t):return dict(shape=list(t.shape),stride=list(t.stride()),offset=t.storage_offset(),dtype=str(t.dtype),device=str(t.device),data_ptr=t.data_ptr(),storage_ptr=t.untyped_storage().data_ptr(),storage_bytes=t.untyped_storage().nbytes())
def digest(t):
 import torch
 return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def dense(t,shape,torch):
 assert isinstance(t,torch.Tensor) and t.dtype==torch.bfloat16 and t.device.type=='cuda'
 assert list(t.shape)==shape and list(t.stride())==[shape[1],1] and t.storage_offset()==0 and t.untyped_storage().nbytes()==2*shape[0]*shape[1]
def census(rows,names):
 return Counter((r['path'],r['phase'])for r in rows)==Counter({(n,p):v for n in names for p,v in [('prefill',1),('decode',15)]})

class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.saved=[];self.others=[];self.modules=[];self.live=[];self.rows=[];self.mode='shadow';self.shadow_pass=False;self.pending=False;self.first_mismatch=None;self.counts={};self.native_selected=0;self.tile_selected=0;self.prefill_native=0;self.shadow_refs={}
 def install(self,model):
  from vllm.model_executor.layers.linear import UnquantizedLinearMethod
  from vllm.model_executor.layers.utils import default_unquantized_gemm
  import torch
  owners=[(n,m,m.quant_method)for n,m in model.named_modules()if getattr(m,'quant_method',None)is not None and callable(getattr(m.quant_method,'apply',None))]
  targets=[]
  for n,m,q in owners:
   match=re.search(r'(?:^|\.)layers\.(\d+)\.linear_attn\.in_proj_ba$',n)
   if match:targets.append((int(match[1]),n,m,q))
  assert len(targets)==24 and sorted(t[0]for t in targets)==list(LAYERS)
  target_ids={id(q)for _,_,_,q in targets};assert len(target_ids)==24 and all(sum(id(q)==id(qq)for _,_,qq in owners)==1 for _,_,_,q in targets)
  for _,n,m,q in targets:
   assert type(q)is UnquantizedLinearMethod and getattr(m,'bias',None)is None and q._gemm_impl is default_unquantized_gemm
   assert 'apply'not in q.__dict__;dense(m.weight,[64,2560],torch)
  self.others=[(q,q.apply)for _,_,q in owners if id(q)not in target_ids]
  try:
   for layer,n,m,q in targets:
    orig=q.apply;self.live.append((n,m,digest(m.weight)));self.modules.append(dict(path=n,layer=layer,weight=metadata(m.weight),weight_sha256=digest(m.weight),apply_source=source(orig),gemm_source=source(q._gemm_impl)))
    def apply(this,owner,x,bias=None,_orig=orig,_m=m,_name=n):
     assert owner is _m and bias is None and not self.pending
     idx=self.counts.get(_name,0);assert idx<16;phase='prefill'if idx==0 else'decode';self.counts[_name]=idx+1
     dense(x,[1696 if phase=='prefill'else 32,2560],torch);dense(owner.weight,[64,2560],torch)
     if phase=='prefill':
      y=_orig(owner,x,bias);self.prefill_native+=1
      self.rows.append(dict(path=_name,phase=phase,mode=self.mode,passed=True,original_calls=1,tileir_calls=0,scope='Unchanged original call, not candidate-qualified'))
      return y
     from packed_k128_partial_v1.api import linear,selected_path
     from packed_contiguous_split2_qualification_v1.run import compare,passing
     assert selected_path(*[32,64,2560])=='bf16_k128'
     self.pending=True;row=dict(path=_name,phase=phase,mode=self.mode,call_index=idx,original_calls=0,tileir_calls=0,candidate_path='bf16_k128');self.rows.append(row)
     try:
      before=(digest(x),digest(owner.weight));row['input_hashes']=list(before);xm,wm=metadata(x),metadata(owner.weight);xd,wd=x.detach(),owner.weight.detach();assert metadata(xd)==xm and metadata(wd)==wm and not xd.requires_grad and not wd.requires_grad
      if self.mode=='shadow':
       native=_orig(owner,x,bias);row['original_calls']=1;self.native_selected+=1;dense(native,[32,64],torch)
       assert (digest(x),digest(owner.weight))==before,'Original mutated input/backing'
      else:
       assert self.mode=='replace'and self.shadow_pass
       expected=self.shadow_refs[(_name,idx)];row['same_shadow_inputs']=list(before)==expected['input_hashes'];assert row['same_shadow_inputs'],'Replacement input differs from verified native trajectory'
      with torch.no_grad():tile=linear(xd,wd)
      self.tile_selected+=1;row['tileir_calls']=1;dense(tile,[32,64],torch)
      row['inputs_preserved']=(digest(x),digest(owner.weight))==before and metadata(x)==xm and metadata(owner.weight)==wm
      row['input_backing_preserved']=row['inputs_preserved'];assert row['inputs_preserved']
      row['output_contract_pass']=tile.untyped_storage().data_ptr() not in (x.untyped_storage().data_ptr(),owner.weight.untyped_storage().data_ptr());assert row['output_contract_pass']
      row['output_finite']=bool(torch.isfinite(tile).all());row['output_sha256']=digest(tile)
      if self.mode=='shadow':
       assert native.untyped_storage().data_ptr()not in (x.untyped_storage().data_ptr(),owner.weight.untyped_storage().data_ptr())
       row['comparison']=compare(native.detach().cpu(),tile.detach().cpu(),torch)
       row['native_output_sha256']=digest(native);row['passed']=passing(row['comparison'])and row['output_finite']
       if not row['passed']:
        bank=self.output/'first_mismatch';bank.mkdir(exist_ok=False);artifacts={}
        for label,t in [('X',x),('W',owner.weight),('nativeY',native),('tileY',tile)]:
         p=bank/(label+'.pt');torch.save(t.detach().cpu().contiguous(),p);artifacts[label]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),metadata=metadata(t))
        self.first_mismatch=dict(path=_name,call_index=idx,artifacts=artifacts,comparison=row['comparison']);self.save();raise RuntimeError('K128 live raw/strict/finite HOLD; replacement not executed')
       returned=native
      else:
       row['same_shadow_output']=row['output_sha256']==expected['native_output_sha256'];row['passed']=row['output_finite']and row['same_shadow_output'];assert row['passed'];returned=tile
      if len(self.rows)%24==0:self.save()
      return returned
     finally:self.pending=False
    bound=types.MethodType(apply,q);self.saved.append((q,orig,bound));q.apply=bound
   return dict(eligible_modules=24,modules=self.modules,other_apply_bindings=len(self.others),other_GEMM_policy='Untouched native public methods')
  except BaseException:self.close();raise
 def unchanged_others(self):return all(q.apply==orig for q,orig in self.others)
 def start(self,mode):
  assert mode in ('shadow','replace')and not self.pending and self.unchanged_others()
  if mode=='replace':assert self.shadow_pass
  self.mode=mode;self.rows=[];self.counts={};self.native_selected=self.tile_selected=self.prefill_native=0
  return dict(mode=mode)
 def summary(self):
  return dict(mode=self.mode,shadow_pass=self.shadow_pass,rows=self.rows,supported_calls=self.tile_selected,tileir_gate_calls=self.tile_selected,native_gate_calls=self.native_selected,prefill_native_calls=self.prefill_native,supported_vendor_fallback_calls=0,unsupported_native_fallback_calls=0,other_apply_bindings_unchanged=self.unchanged_others(),first_mismatch=self.first_mismatch,pending=self.pending)
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2)+'\n')
 def finish(self):
  assert not self.pending and census(self.rows,[n for n,_,_ in self.live])
  assert self.tile_selected==360 and self.prefill_native==24 and self.native_selected==(360 if self.mode=='shadow'else 0)
  assert self.unchanged_others()and self.first_mismatch is None
  assert all(digest(m.weight)==h for _,m,h in self.live)
  assert all(r['passed']for r in self.rows)
  if self.mode=='shadow':
   self.shadow_refs={(r['path'],r['call_index']):dict(input_hashes=r['input_hashes'],native_output_sha256=r['native_output_sha256'])for r in self.rows if r['phase']=='decode'};assert len(self.shadow_refs)==360;self.shadow_pass=True
  self.save();return dict(self.summary(),weights_unchanged=True)
 def close(self):
  conflicts=[]
  for q,orig,bound in reversed(self.saved):
   if q.__dict__.get('apply')is not bound:conflicts.append(type(q).__name__);continue
   delattr(q,'apply');assert q.apply==orig
  self.saved=[];assert self.unchanged_others()and not conflicts,conflicts
  return dict(restored=True,other_apply_bindings_unchanged=True)
def install(model,output):
 assert not hasattr(model,ATTR);a=Adapter(output);setattr(model,ATTR,a)
 try:return a.install(model)
 except BaseException:a.close();delattr(model,ATTR);raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True,other_apply_bindings_unchanged=True)
 r=a.close();delattr(model,ATTR);return r
