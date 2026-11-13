"""Compose unchanged numerical kernels at public eager boundaries; no GPU at import."""
from pathlib import Path
from public_vllm_composed_integration_v11.adapter import PARTS as SIXTEEN
from all_gemm_runtime_integration_v1 import adapter as gemm
from sampler_three_ordered_runtime_integration_v2 import adapter as top5
ATTR='_all_gemm_sixteen_sampler_v2'
PARTS=(*SIXTEEN,('all_gemm',gemm,2448),('sampler_three',top5,16))
ORDER=tuple(x[0] for x in PARTS)
# These legacy shadow wrappers save a failed row but return native until finish.
# Preserve their diagnostic save, then raise immediately; numerical code is unchanged.
SAVE_GUARDED=frozenset(('norm','gated_norm','indexed_decode_conv'))
class Controller:
 def __init__(self,model,root,parts=PARTS):
  self.model=model;self.root=Path(root);self.parts=parts;self.installed=[];self.guards=[];self.failed=None;self.mode=None;self.active=False;self.shadow_pass=False
 def latch(self,key,error):
  if self.failed is None:self.failed=dict(component=key,error=repr(error))
 def healthy(self):
  assert self.failed is None,('Composition latched',self.failed)
  for key,module in self.installed:
   obj=getattr(self.model,module.ATTR,None)
   if obj is None:continue
   rows=getattr(obj,'rows',[])
   bad=getattr(obj,'first_mismatch',None) or getattr(obj,'failed',None) or (rows and rows[-1].get('passed') is False)
   if bad:
    e=RuntimeError('Component failure latch: '+key);self.latch(key,e);raise e
 def guard_save(self,key,module):
  if key not in SAVE_GUARDED:return
  obj=getattr(self.model,module.ATTR);old=obj.save;had='save' in obj.__dict__;prior=obj.__dict__.get('save')
  def save(*args,**kwargs):
   try:
    ret=old(*args,**kwargs);self.healthy();return ret
   except BaseException as e:self.latch(key,e);raise
  self.guards.append((obj,had,prior,save));obj.save=save
 def install(self):
  out={}
  for key,module,count in self.parts:
   self.installed.append((key,module)) # cleanup includes partially installed module
   out[key]=module.install(self.model,str(self.root/key));self.guard_save(key,module)
  self.healthy();return dict(components=out,install_order=[k for k,m in self.installed],controller_count=18,selected_boundary_calls_expected=5648,selected_logical_operations_expected=5680,scope='16 existing functional adapters + all13 GEMM contracts + Top5/ranks/selected-logprobs; native cat/attention core/GDN prefill retained')
 def start(self,mode):
  self.healthy();assert not self.active and mode in ('shadow','replace')
  assert (mode=='shadow' and self.mode is None) or (mode=='replace' and self.mode=='shadow' and self.shadow_pass)
  self.mode=mode;self.active=True
  try:return {k:m.start(self.model,mode) for k,m,n in self.parts}
  except BaseException as e:self.latch('start',e);raise
 def finish(self):
  self.healthy();assert self.active
  try:
   components={k:m.finish(self.model) for k,m,n in self.parts};self.healthy()
   for k,m,n in self.parts:
    d=components[k];assert d['selected_calls' if k=='all_gemm' else 'supported_calls']==n,(k,n)
    assert d['shadow_pass'],k
   self.active=False
   if self.mode=='shadow':self.shadow_pass=True
   return dict(components=components,shadow_pass=self.shadow_pass,failed=self.failed,selected_boundary_calls=5648,selected_logical_operations=5680,selected_vendor_fallback_calls=sum(d.get('selected_vendor_fallback_calls',d.get('supported_vendor_fallback_calls',0)) for d in components.values()),installation_order=list(ORDER),convolution_scope='decode prefill counter24 delegates to prefill TileIR; not native fallback',weight_validation_scope='GEMM W four full-byte phase boundaries, per-call metadata/version; not per-call W bytes')
  except BaseException as e:self.latch('finish',e);raise
 def close(self):
  results={};errors=[]
  # Restore save callbacks before adapter cleanup can save failure evidence again.
  for obj,had,prior,guard in reversed(self.guards):
   try:
    assert obj.__dict__.get('save')is guard
    if had:obj.save=prior
    else:delattr(obj,'save')
   except BaseException as e:errors.append(('save_guard',repr(e)))
  self.guards=[]
  for key,module in reversed(self.installed):
   try:results[key]=module.close(self.model)
   except BaseException as e:errors.append((key,repr(e)))
  order=[k for k,m in reversed(self.installed)];self.installed=[]
  assert not errors,errors
  assert all(v['restored'] for v in results.values())
  return dict(restored=True,components=results,close_order=order,save_guards_restored=True)
def install(model,output):
 assert not hasattr(model,ATTR);root=Path(output);root.mkdir(exist_ok=False);a=Controller(model,root);setattr(model,ATTR,a)
 try:return a.install()
 except BaseException:
  try:a.close()
  finally:delattr(model,ATTR)
  raise
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True,already_absent=True)
 try:return a.close()
 finally:delattr(model,ATTR)
