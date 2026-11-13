"""Process-local public apply adapter for153 observed GEMM owners only."""
from pathlib import Path
import hashlib,inspect,json,types
from .contract import plan,phase_shape,coverage
from .metrics import digest,compare,passing
ATTR='_all_gemm_runtime_integration_v1'

def version(t):
 try:return t._version
 except RuntimeError as e:
  if 'Inference tensors do not track version counter'not in str(e):raise
  return 'unavailable_inference_tensor'

def metadata(t):return dict(shape=list(t.shape),stride=list(t.stride()),offset=t.storage_offset(),dtype=str(t.dtype),device=str(t.device),data_ptr=t.data_ptr(),storage_ptr=t.untyped_storage().data_ptr(),storage_bytes=t.untyped_storage().nbytes(),version=version(t))
def dense(t,shape,torch):
 assert isinstance(t,torch.Tensor),('Not Tensor',type(t))
 assert t.dtype==torch.bfloat16 and t.device.type=='cuda',('dtype/device',str(t.dtype),str(t.device))
 assert list(t.shape)==shape and list(t.stride())==[shape[1],1]and t.storage_offset()==0 and t.untyped_storage().nbytes()==shape[0]*shape[1]*2,('Unsupported dense view',shape,metadata(t))

def source(fn):
 fn=getattr(fn,'__func__',fn);p=inspect.getsourcefile(fn)
 return dict(module=fn.__module__,qualname=fn.__qualname__,path=p,sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())

class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.plan=plan();self.saved=[];self.live={};self.unique={};self.modules=[];self.counts={};self.rows=[];self.mode=None;self.failed=None;self.pending=False;self.phase_active=False;self.shadow_pass=False;self.shadow_refs={};self.boundaries={};self.initial_weights=None;self.excluded_calls=0;self.native_calls=0;self.tile_calls=0;self.k128_calls=0;self.base_calls=0
 def latch(self,error,context):
  if self.failed is None:self.failed=dict(error=repr(error),context=context)
  try:self.save()
  except BaseException:pass
 def enter(self,context):
  if self.failed is not None:raise RuntimeError('Prior failure latched; no further native/candidate entry')
  if self.pending or not self.phase_active:
   error=RuntimeError('Reentrant or out-of-phase apply');self.latch(error,context);raise error
 def stable_weight(self,name):
  m,q,w,meta,key=self.live[name]
  assert m.quant_method is q and m.weight is w and metadata(w)==meta,('Weight owner/storage/version changed',name)
  return key
 def scan_weights(self,tag):
  import torch
  records={}
  for name in self.live:self.stable_weight(name)
  for key,(w,meta,owners)in self.unique.items():
   assert metadata(w)==meta;v=digest(w,torch);assert v['finite'];records[key]=dict(v,owners=owners,metadata=meta)
  hashes={k:v['sha256']for k,v in records.items()}
  if self.initial_weights is None:self.initial_weights=hashes
  assert hashes==self.initial_weights,'Full weight bytes changed across phase boundary'
  self.boundaries[tag]=dict(records=records,unique_weights=len(records),full_bytes=sum(v['bytes']for v in records.values()))
  return records
 def install(self,model):
  import torch
  from vllm.model_executor.layers.linear import UnquantizedLinearMethod
  from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod
  from vllm.model_executor.layers.utils import default_unquantized_gemm,dispatch_unquantized_gemm
  from vllm import envs
  owners={n:(m,m.quant_method)for n,m in model.named_modules()if getattr(m,'quant_method',None)is not None and callable(getattr(m.quant_method,'apply',None))}
  expected={x['path']:x for x in self.plan['selected']};excluded={x['path']:x for x in self.plan['excluded']}
  assert set(owners)==set(expected)|set(excluded)and len({id(q)for m,q in owners.values()})==178
  assert not envs.VLLM_BATCH_INVARIANT and dispatch_unquantized_gemm()is default_unquantized_gemm
  self.native_dispatch=default_unquantized_gemm
  keys={}
  for name,spec in expected.items():
   m,q=owners[name];assert getattr(m,'bias',None)is None and 'apply'not in q.__dict__;dense(m.weight,spec['weight_shape'],torch)
   assert type(q)is(UnquantizedEmbeddingMethod if spec['kind']=='head'else UnquantizedLinearMethod)
   if spec['kind']!='head':assert q._gemm_impl is default_unquantized_gemm
   orig=q.apply;ident=source(orig);assert {k:ident[k]for k in ('module','qualname')}==spec['apply_source']
   meta=metadata(m.weight);finger=(meta['storage_ptr'],meta['storage_bytes'],meta['offset'],tuple(meta['shape']),tuple(meta['stride']),meta['dtype'],meta['device'])
   if finger not in keys:
    key='weight'+str(len(keys));keys[finger]=key;self.unique[key]=(m.weight,meta,[])
   key=keys[finger];self.unique[key][2].append(name);self.live[name]=(m,q,m.weight,meta,key)
   self.modules.append(dict(path=name,kind=spec['kind'],weight_key=key,weight=meta,apply_source=ident,version_check_available=meta['version']!='unavailable_inference_tensor'))
  for name,spec in excluded.items():
   m,q=owners[name];assert list(m.weight.shape)==spec['weight_shape']and'apply'not in q.__dict__
  try:
   for name,(m,q)in owners.items():
    orig=q.apply
    if name in excluded:
     def guarded(this,*args,_name=name,**kwargs):
      self.enter(dict(path=_name,excluded=True));self.excluded_calls+=1;e=RuntimeError('Excluded non-GEMM apply unexpectedly invoked');self.latch(e,dict(path=_name));raise e
     bound=types.MethodType(guarded,q)
    else:
     spec=expected[name]
     def apply(this,owner,x,bias=None,_orig=orig,_m=m,_name=name,_spec=spec):
      self.enter(dict(path=_name));self.pending=True
      try:
       assert owner is _m and bias is None and this is owner.quant_method
       idx=self.counts.get(_name,0);phase,(M,N,K)=phase_shape(_spec,idx);self.counts[_name]=idx+1;dense(x,[M,K],torch);key=self.stable_weight(_name)
       if _spec['kind']=='head':assert not envs.VLLM_BATCH_INVARIANT and dispatch_unquantized_gemm()is self.native_dispatch
       else:assert this._gemm_impl is self.native_dispatch
       row=dict(path=_name,kind=_spec['kind'],phase=phase,call_index=idx,logical_MNK=[M,N,K],mode=self.mode,weight_key=key,weight_metadata_match=True,weight_version_available=self.live[_name][3]['version']!='unavailable_inference_tensor',native_calls=0,tileir_calls=0);self.rows.append(row)
       xm=metadata(x);xd=x.detach();wd=owner.weight.detach();assert metadata(xd)==xm and metadata(wd)==metadata(owner.weight)and not xd.requires_grad and not wd.requires_grad
       before=digest(x,torch);assert before['finite'];row['input_sha256']=before['sha256']
       if self.mode=='shadow':
        native=_orig(owner,x,bias);self.native_calls+=1;row['native_calls']=1;dense(native,[M,N],torch);self.stable_weight(_name)
        assert digest(x,torch)['sha256']==before['sha256']and metadata(x)==xm,'Native changed X bytes/metadata'
       else:
        assert self.mode=='replace'and self.shadow_pass;prior=self.shadow_refs[(_name,idx)]
        row['same_shadow_inputs']=before['sha256']==prior['input_sha256']and key==prior['weight_key'];assert row['same_shadow_inputs']
       if _spec['kind']=='ba'and phase=='decode':
        from packed_k128_partial_v1 import linear
        row['candidate_backend']='K128';self.k128_calls+=1
       else:
        from tileirblas_once_occ2 import linear
        row['candidate_backend']='existing';self.base_calls+=1
       with torch.no_grad():tile=linear(xd,wd)
       self.tile_calls+=1;row['tileir_calls']=1;dense(tile,[M,N],torch);self.stable_weight(_name)
       row['input_preserved']=digest(x,torch)['sha256']==before['sha256']and metadata(x)==xm;assert row['input_preserved']
       row['output_contract_pass']=tile.untyped_storage().data_ptr()not in(x.untyped_storage().data_ptr(),owner.weight.untyped_storage().data_ptr());assert row['output_contract_pass']
       if self.mode=='shadow':
        assert native.untyped_storage().data_ptr()not in(x.untyped_storage().data_ptr(),owner.weight.untyped_storage().data_ptr())
        c=compare(native,tile,torch);row['comparison']=c;row['native_output_sha256']=c['native_sha256'];row['output_sha256']=c['tile_sha256'];row['passed']=passing(c)
        if not row['passed']:
         self.save_failure(row,x,native,tile,torch);raise RuntimeError('Full GEMM raw/strict/finite HOLD; replacement forbidden')
        returned=native
       else:
        out=digest(tile,torch);row['output_sha256']=out['sha256'];row['output_finite']=out['finite'];row['same_shadow_output']=out['sha256']==prior['native_output_sha256'];row['passed']=out['finite']and row['same_shadow_output'];assert row['passed'];returned=tile
       if len(self.rows)%153==0:self.save()
       return returned
      except BaseException as e:self.latch(e,dict(path=_name,call_index=self.counts.get(_name)));raise
      finally:self.pending=False
     bound=types.MethodType(apply,q)
    self.saved.append((q,orig,bound));q.apply=bound
   return dict(selected_modules=153,excluded_guarded_modules=25,modules=self.modules,weight_validation_scope='phase-boundary full bytes plus per-call owner/storage/metadata/version when available; NOT per-call W bytes')
  except BaseException:self.close();raise
 def save_failure(self,row,x,native,tile,torch):
  amount=sum(t.numel()*t.element_size()for t in(x,native,tile));assert amount<=128*1024*1024
  bank=self.output/'first_mismatch';bank.mkdir(exist_ok=False);art={}
  for name,t in [('X',x),('nativeY',native),('tileY',tile)]:
   p=bank/(name+'.pt');torch.save(t.detach().cpu().contiguous(),p);art[name]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
  row['failure_artifacts']=art;row['weight_evidence']='phase boundary digest and current metadata; no new weight dump'
 def _start(self,mode):
  assert mode in('shadow','replace')and self.failed is None and not self.pending and not self.phase_active
  if mode=='replace':assert self.shadow_pass
  self.mode=mode;self.rows=[];self.counts={};self.native_calls=self.tile_calls=self.k128_calls=self.base_calls=0
  self.scan_weights(mode+'_start');self.phase_active=True;return dict(mode=mode)
 def start(self,mode):
  try:return self._start(mode)
  except BaseException as e:self.latch(e,dict(stage='phase_start',mode=mode));raise
 def summary(self):
  return dict(mode=self.mode,rows=self.rows,shadow_pass=self.shadow_pass,failed=self.failed,pending=self.pending,selected_calls=self.tile_calls,native_calls=self.native_calls,tileir_calls=self.tile_calls,k128_calls=self.k128_calls,existing_calls=self.base_calls,excluded_apply_calls=self.excluded_calls,selected_vendor_fallback_calls=0,weight_boundaries=self.boundaries,weight_validation_scope='phase boundaries full bytes; per-call metadata/version only, no per-call W byte claim')
 def save(self):
  if self.mode is not None:(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2)+'\n')
 def _finish(self):
  assert self.failed is None and not self.pending and self.phase_active and coverage(self.rows,self.plan)
  assert self.tile_calls==2448 and self.k128_calls==360 and self.base_calls==2088 and self.native_calls==(2448 if self.mode=='shadow'else 0)and self.excluded_calls==0
  self.scan_weights(self.mode+'_end');self.phase_active=False;assert all(r['passed']for r in self.rows)
  if self.mode=='shadow':
   self.shadow_refs={(r['path'],r['call_index']):dict(input_sha256=r['input_sha256'],weight_key=r['weight_key'],native_output_sha256=r['native_output_sha256'])for r in self.rows};assert len(self.shadow_refs)==2448;self.shadow_pass=True
  self.save();return self.summary()
 def finish(self):
  try:return self._finish()
  except BaseException as e:self.latch(e,dict(stage='phase_finish',mode=self.mode));raise
 def close(self):
  conflicts=[]
  for q,orig,bound in reversed(self.saved):
   if q.__dict__.get('apply')is not bound:conflicts.append(type(q).__name__);continue
   try:
    delattr(q,'apply')
    if q.apply!=orig:conflicts.append(type(q).__name__+':restore mismatch')
   except BaseException as e:conflicts.append(repr(e))
  self.saved=[]
  if self.phase_active:
   try:self.scan_weights(self.mode+'_cleanup_end')
   finally:self.phase_active=False
  assert not conflicts,conflicts
  self.save();return dict(restored=True,conflicts=[])
def install(model,output):
 assert not hasattr(model,ATTR);a=Adapter(output);setattr(model,ATTR,a)
 try:return a.install(model)
 except BaseException:a.close();delattr(model,ATTR);raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True,already_absent=True)
 r=a.close();delattr(model,ATTR);return r
