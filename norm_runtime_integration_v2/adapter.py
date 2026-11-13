"""Process-local GemmaRMSNorm forward adapter; no linear/GEMM patch."""
import types,json,hashlib,re
from collections import Counter
from pathlib import Path
from norm_runtime_qualification_v2.run import logical,descriptor,aliases,identity
from norm_runtime_qualification_v2.compare import metrics
ATTR='_tileir_norm_runtime_integration_v2'
def operand_metadata(t):
 return dict(descriptor(t),requires_grad=bool(t.requires_grad),tensor_class=type(t).__module__+'.'+type(t).__qualname__,layout=str(t.layout),data_ptr=t.data_ptr(),storage_ptr=t.untyped_storage().data_ptr(),storage_nbytes=t.untyped_storage().nbytes())
def verify_detach(original,candidate):
 a,b=operand_metadata(original),operand_metadata(candidate)
 assert b['requires_grad'] is False
 for key in ('shape','stride','storage_offset','dtype','device','layout','data_ptr','storage_ptr','storage_nbytes'):assert a[key]==b[key],('detach altered view',key)
 return dict(original=a,candidate=b,same_storage_view=True)
def new_dispatch_observer():
 from vllm import ir
 from norm_runtime_contract_capture_v2.dispatch_observer import DispatchObserver
 return DispatchObserver({'rms_norm':ir.ops.rms_norm,'fused_add_rms_norm':ir.ops.fused_add_rms_norm},identity).install()
class Adapter:
 def __init__(self,output):self.output=Path(output);self.output.mkdir(exist_ok=False);self.saved=[];self.mode='shadow';self.rows=[];self.unsupported=Counter();self.installed=[];self.shadow_pass=False;self.first_mismatch=None;self.dispatch=None;self.current_call=None
 def install(self,model):
  found=[(n,m) for n,m in model.named_modules() if type(m).__name__=='GemmaRMSNorm']
  eligible=[(n,m) for n,m in found if tuple(m.weight.shape)==(2560,)]
  assert len(eligible)==65,('Expected32 input+32 post+final',len(eligible))
  layer_keys=[];final_names=[]
  for n,m in eligible:
   match=re.search(r'(?:^|\.)layers\.(\d+)\.(input_layernorm|post_attention_layernorm)$',n)
   if match:layer_keys.append((int(match[1]),match[2]))
   else:assert n=='norm' or n.endswith('.norm');final_names.append(n)
  assert set(layer_keys)=={(i,k) for i in range(32) for k in ('input_layernorm','post_attention_layernorm')} and len(layer_keys)==64 and len(final_names)==1
  try:
   self.dispatch=new_dispatch_observer()
   for name,m in found:
    supported=tuple(m.weight.shape)==(2560,);orig=m.forward;had='forward' in m.__dict__;prior=m.__dict__.get('forward')
    self.installed.append(dict(path=name,supported=supported,weight_shape=list(m.weight.shape),original_weight_metadata=operand_metadata(m.weight),forward=identity(orig),selected_forward=identity(getattr(m,'_forward_method',orig))))
    def forward(this,x,residual=None,_orig=orig,_name=name,_supported=supported):
     if not _supported:
      self.unsupported[str((_name,tuple(x.shape)))]+=1;return _orig(x,residual)
     from tileir_vllm_gemma_norm_v1 import gemma_rms_norm,validate
     import torch
     candidate_x=x.detach();candidate_w=this.weight.detach();candidate_r=None if residual is None else residual.detach()
     detach_checks={'x':verify_detach(x,candidate_x),'weight':verify_detach(this.weight,candidate_w)}
     if residual is not None:detach_checks['residual']=verify_detach(residual,candidate_r)
     self.current_call=dict(path=_name,mode=self.mode,call_index=len(self.rows),detach_checks=detach_checks)
     try:validate(candidate_x,candidate_w,this.variance_epsilon,candidate_r)
     except BaseException:self.save();raise
     vals={'x':x,'weight':this.weight}
     if residual is not None:vals['residual']=residual
     if self.mode=='replace':
      out=gemma_rms_norm(candidate_x,candidate_w,this.variance_epsilon,candidate_r)
      self.rows.append(dict(path=_name,shape=list(x.shape),residual=residual is not None,mode='replace',detach_checks=detach_checks));self.current_call=None;return out
     assert self.mode=='shadow'
     before={k:logical(v,torch) for k,v in vals.items()}
     dispatch_key=str(len(self.rows));self.dispatch.enter(dispatch_key)
     try:native=_orig(x,residual)
     finally:provider_events=self.dispatch.leave(dispatch_key)
     after_native={k:logical(v,torch) for k,v in vals.items()}
     # Do not feed mutated inputs into a purported SAME-input comparison.
     assert before==after_native,'Original norm mutated inputs; functional contract unsupported'
     with torch.no_grad():tile=gemma_rms_norm(candidate_x,candidate_w,this.variance_epsilon,candidate_r)
     ns=list(native) if isinstance(native,tuple) else [native];ts=list(tile) if isinstance(tile,tuple) else [tile]
     assert len(ns)==len(ts)
     comparisons=[metrics(t,n) for t,n in zip(ts,ns)]
     nv=dict(vals,**{'output'+str(i):v for i,v in enumerate(ns)});tv=dict(vals,**{'output'+str(i):v for i,v in enumerate(ts)})
     layout_equal=all(all(descriptor(a)[k]==descriptor(b)[k] for k in ('shape','stride','storage_offset','dtype')) for a,b in zip(ts,ns))
     row=dict(path=_name,shape=list(x.shape),residual=residual is not None,mode='shadow',detach_checks=detach_checks,provider_events=provider_events,comparisons=comparisons,inputs_preserved=all(logical(v,torch)==before[k] for k,v in vals.items()),alias_equal=aliases(nv)==aliases(tv),layout_equal=layout_equal,return_kind_equal=isinstance(native,tuple)==isinstance(tile,tuple))
     row['provider_contract_pass']=len(provider_events)==1 and provider_events[0]['op']==('rms_norm' if residual is None else 'fused_add_rms_norm') and provider_events[0]['provider']=='native' and provider_events[0]['inplace'] is False and provider_events[0]['supported'] is True
     row['passed']=row['provider_contract_pass'] and row['inputs_preserved'] and row['alias_equal'] and row['layout_equal'] and row['return_kind_equal'] and all(c['raw_equal'] and c['candidate_nonfinite']==c['reference_nonfinite']==0 for c in comparisons)
     if not row['passed'] and self.first_mismatch is None:
      bank=self.output/'first_mismatch';bank.mkdir(exist_ok=False);artifacts={}
      tensors=dict(vals,**{'nativeY'+str(i):v for i,v in enumerate(ns)},**{'tileY'+str(i):v for i,v in enumerate(ts)})
      for key,value in tensors.items():
       path=bank/(key+'.pt');torch.save(value.detach().cpu().contiguous(),path)
       artifacts[key]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),original_descriptor=descriptor(value))
      self.first_mismatch=dict(call_index=len(self.rows),path=_name,artifacts=artifacts,comparison=row)
     self.rows.append(row);self.current_call=None
     if not row['passed'] or len(self.rows)%32==0:self.save()
     return native
    bound=types.MethodType(forward,m);self.saved.append((m,had,prior,bound));m.forward=bound
   return dict(modules=self.installed,eligible_modules=65,other_gemma_modules=len(found)-65)
  except BaseException:self.close();raise
 def start(self,mode):
  assert mode in ('shadow','replace')
  if mode=='replace':assert self.shadow_pass,'Replacement requires complete raw shadow PASS'
  self.mode=mode;self.rows=[];self.unsupported.clear();return dict(mode=mode)
 def summary(self):return dict(current_call=self.current_call,supported_vendor_fallback_calls=0,unsupported_native_calls=sum(self.unsupported.values()),first_mismatch=self.first_mismatch,mode=self.mode,rows=self.rows,unsupported_calls=dict(self.unsupported),supported_calls=len(self.rows),shadow_pass=self.shadow_pass,scope='All width2560 Gemma modules only; other Gemma calls remain native and counted')
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2)+'\n')
 def finish(self):
  supported=[x['path'] for x in self.installed if x['supported']]
  counts=Counter((x['path'],tuple(x['shape'])) for x in self.rows)
  assert counts==Counter({(n,(m,2560)):k for n in supported for m,k in [(1696,1),(32,15)]}),dict(counts)
  if self.mode=='shadow':self.shadow_pass=all(x['passed'] for x in self.rows)
  self.save();r=self.summary();r['counts']=[dict(path=k[0],shape=list(k[1]),calls=v) for k,v in sorted(counts.items())];return r
 def close(self):
  conflicts=[]
  for m,had,prior,bound in reversed(self.saved):
   if m.__dict__.get('forward') is not bound:conflicts.append(type(m).__name__);continue
   if had:m.forward=prior
   else:delattr(m,'forward')
  self.saved=[]
  if self.dispatch is not None:self.dispatch.close();self.dispatch=None
  assert not conflicts,conflicts;return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR);a=Adapter(output);result=a.install(model);setattr(model,ATTR,a);return result
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 result=a.close();delattr(model,ATTR);return result
