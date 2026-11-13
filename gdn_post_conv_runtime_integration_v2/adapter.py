"""Public helper functional replacement, native GDN/conv/GEMM otherwise untouched."""
from pathlib import Path
import json,hashlib,inspect,functools
from collections import Counter
ATTR='_tileir_post_conv_integration_v1'
INPUTS=('conv_output','a','b','A_log','dt_bias');OUTPUTS=('q','k','v','g','beta')
def candidate_views(values):
 """Inference-only detached views. Original caller/native arguments stay untouched."""
 import torch
 out=dict(values);record={}
 for k in INPUTS:
  original=values[k];assert isinstance(original,torch.Tensor)
  view=original.detach()
  same=(view.untyped_storage().data_ptr()==original.untyped_storage().data_ptr() and view.untyped_storage().nbytes()==original.untyped_storage().nbytes() and view.data_ptr()==original.data_ptr() and view.shape==original.shape and view.stride()==original.stride() and view.storage_offset()==original.storage_offset() and view.dtype==original.dtype and view.device==original.device)
  assert same and not view.requires_grad
  record[k]=dict(original_type=type(original).__module__+'.'+type(original).__name__,original_requires_grad=original.requires_grad,candidate_type=type(view).__module__+'.'+type(view).__name__,candidate_requires_grad=view.requires_grad,same_storage_layout_dtype=same)
  out[k]=view
 return out,record
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.mode='shadow';self.shadow_pass=False;self.rows=[];self.live={};self.bindings=[];self.native_gate_calls=0;self.tileir_gate_calls=0;self.first_mismatch=None
 def install(self,model):
  import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q
  from vllm.forward_context import get_forward_context
  for n,m in model.named_modules():
   if n.endswith('.linear_attn') and hasattr(m,'A_log') and hasattr(m,'dt_bias'):self.live[n]=m
  assert len(self.live)==24
  original=q.fused_post_conv_prep;sig=inspect.signature(original)
  @functools.wraps(original)
  def wrapper(*args,**kwargs):
   bound=sig.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments
   owners=[(n,m) for n,m in self.live.items() if v['A_log'] is m.A_log and v['dt_bias'] is m.dt_bias];assert len(owners)==1
   name,m=owners[0];ctx=get_forward_context();assert isinstance(ctx.attn_metadata,dict)
   from gdn_post_conv_runtime_capture_v1.observer import phase_meta
   phase=phase_meta(ctx.attn_metadata.get(m.prefix));assert phase['phase']=='prefill' and not phase['speculative']
   return self.apply(name,phase,v,lambda:original(*args,**kwargs))
  q.fused_post_conv_prep=wrapper;self.bindings.append((q,'fused_post_conv_prep',original,wrapper))
  return dict(modules=list(self.live),count=24,boundary='public imported fused_post_conv_prep',scope='eager prefill helper only')
 def apply(self,name,phase,v,native):
  import torch
  from tileir_vllm_post_conv_v2.api import fused_post_conv_prep,validate
  from gdn_post_conv_runtime_capture_v1.observer import fullbytes
  from norm_runtime_contract_capture_v1.observer import metadata,aliases
  from gdn_post_conv_qualification_v2.run import compare
  assert self.mode in ('shadow','replace') and (self.mode!='replace' or self.shadow_pass)
  candidate_args,detach_record=candidate_views(v)
  validate(**candidate_args)
  before={k:fullbytes(v[k]) for k in INPUTS};descriptors={k:metadata(v[k]) for k in INPUTS}
  finite=all(bool(torch.isfinite(v[k]).all()) for k in INPUTS);assert finite,'Nonfinite input HOLD'
  if self.mode=='shadow':
   original=native();self.native_gate_calls+=1
   assert isinstance(original,tuple) and len(original)==5
   candidate=fused_post_conv_prep(**candidate_args);self.tileir_gate_calls+=1
  else:
   original=None;candidate=fused_post_conv_prep(**candidate_args);self.tileir_gate_calls+=1
  assert isinstance(candidate,tuple) and len(candidate)==5
  preserved={k:torch.equal(fullbytes(v[k]),before[k]) for k in INPUTS}
  layout={};comparisons={};alias_ok=True;outs={}
  for group,values in [('candidate',candidate)]+([('native',original)] if original is not None else []):
   named=dict(zip(OUTPUTS,values));am=aliases(dict({k:v[k] for k in INPUTS},**named));alias_ok &= not any(x['storage_overlap'] for k,x in am.items() if set(k.split('|'))&set(OUTPUTS));outs[group]={'descriptors':{k:metadata(t) for k,t in named.items()},'aliases':am}
  for k,t in zip(OUTPUTS,candidate):
   shape=(1696,16,128) if k in ('q','k') else (1696,32,128) if k=='v' else (1696,32)
   dtype=torch.bfloat16 if k in ('q','k','v') else torch.float32
   layout[k]=tuple(t.shape)==shape and t.dtype==dtype and t.is_contiguous() and t.storage_offset()==0
   if original is not None:
    n=original[OUTPUTS.index(k)];layout[k] &= n.shape==t.shape and n.stride()==t.stride() and n.dtype==t.dtype and n.storage_offset()==t.storage_offset() and n.untyped_storage().nbytes()==t.untyped_storage().nbytes()
    comparisons[k]=compare(n.detach().cpu(),t.detach().cpu(),torch)
   else:comparisons[k]={'raw_equal':None,'strict_failures':None,'candidate_nonfinite':int((~torch.isfinite(t)).sum()),'reference':'prior24-call shadow plus same-run public token/logprob gates; native not called'}
  raw=all(c['raw_equal'] for c in comparisons.values()) if original is not None else None
  out_finite=all(c['candidate_nonfinite']==0 and c.get('reference_nonfinite',0)==0 for c in comparisons.values())
  passed=(raw is not False) and all(preserved.values()) and all(layout.values()) and alias_ok and out_finite
  row=dict(candidate_detach=detach_record,path=name,phase='prefill',actual_phase=phase,mode=self.mode,input_descriptors=descriptors,outputs=outs,comparisons=comparisons,raw_equal=raw,input_fullstorage_preserved=preserved,input_finite=finite,output_finite=out_finite,layout=layout,output_alias_disjoint=alias_ok,return_tuple=True,passed=passed)
  self.rows.append(row)
  if not passed and self.first_mismatch is None:
   bank=self.output/'first_mismatch';bank.mkdir();art={}
   tensors={**{'input_storage_before_'+k:t for k,t in before.items()},**{'input_after_'+k:v[k] for k in INPUTS},**{'candidate_'+k:t for k,t in zip(OUTPUTS,candidate)}}
   if original is not None:tensors.update({'native_'+k:t for k,t in zip(OUTPUTS,original)})
   for k,t in tensors.items():
    p=bank/(k+'.pt');torch.save(t.detach().cpu(),p);art[k]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
   self.first_mismatch=dict(call_index=len(self.rows)-1,row=row,artifacts=art)
  self.save();assert passed,'Post-conv fixed raw/finite/input/layout HOLD'
  return original if original is not None else candidate
 def start(self,mode):
  assert mode in ('shadow','replace') and (mode!='replace' or self.shadow_pass)
  self.mode=mode;self.rows=[];self.native_gate_calls=self.tileir_gate_calls=0;return dict(mode=mode)
 def summary(self):return dict(mode=self.mode,rows=self.rows,supported_calls=len(self.rows),native_gate_calls=self.native_gate_calls,tileir_gate_calls=self.tileir_gate_calls,shadow_pass=self.shadow_pass,first_mismatch=self.first_mismatch,supported_vendor_fallback_calls=0)
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2))
 def finish(self):
  assert Counter(r['path'] for r in self.rows)==Counter({n:1 for n in self.live})
  assert len(self.rows)==24 and all(r['passed'] for r in self.rows) and self.tileir_gate_calls==24 and self.native_gate_calls==(24 if self.mode=='shadow' else 0)
  if self.mode=='shadow':self.shadow_pass=True
  self.save();return self.summary()
 def close(self):
  conflicts=[]
  for q,k,old,new in reversed(self.bindings):
   if getattr(q,k) is new:setattr(q,k,old)
   else:conflicts.append(k)
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
