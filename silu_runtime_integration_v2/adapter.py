"""Replace only each eager MLP SiluAndMul.op boundary, not model forward code."""
import json,re,hashlib
from pathlib import Path
from collections import Counter
ATTR='_tileir_silu_integration_v1'
def overlap(a,b):
 x,y=a.untyped_storage(),b.untyped_storage();return x.data_ptr()<y.data_ptr()+y.nbytes() and y.data_ptr()<x.data_ptr()+x.nbytes()
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.bindings=[];self.modules=[];self.mode='shadow';self.rows=[];self.shadow_pass=False;self.native_gate_calls=0;self.tileir_gate_calls=0;self.first_mismatch=None
 def install(self,model):
  import torch
  from vllm.model_executor.layers.activation import SiluAndMul
  found=[(n,m) for n,m in model.named_modules() if re.search(r'(?:^|\.)layers\.\d+\.mlp\.act_fn$',n)]
  assert len(found)==32 and {int(re.search(r'layers\.(\d+)',n)[1]) for n,m in found}==set(range(32))
  try:
   for name,m in found:
    assert type(m) is SiluAndMul and m.op is torch.ops._C.silu_and_mul
    assert getattr(m._forward_method,'__func__',None) is SiluAndMul.forward_cuda,'Not public eager CUDA op dispatch; no bypass'
    original=m.op;had='op' in m.__dict__;prior=m.__dict__.get('op')
    def wrapper(out,x,_name=name,_original=original):return self.apply(_name,out,x,_original)
    self.bindings.append((m,had,prior,wrapper));m.op=wrapper;self.modules.append(name)
   return dict(modules=self.modules,count=32,boundary='SiluAndMul.forward_cuda -> instance.op(out,x)',scope='eager public custom op; not optimized fused implementation')
  except BaseException:self.close();raise
 def apply(self,name,out,x,native):
  import torch
  from tileir_vllm_silu_mul_v2.api import silu_and_mul,validate
  validate(out,x);assert self.mode in ('shadow','replace') and (self.mode!='replace' or self.shadow_pass)
  before=x.detach().clone();input_finite=bool(torch.isfinite(x).all().item());assert input_finite,'Nonfinite actual input outside fixed finite accuracy gate'
  if self.mode=='shadow':
   nr=native(out,x);self.native_gate_calls+=1;assert nr is None
   candidate=torch.empty_like(out);cr=silu_and_mul(candidate,x);self.tileir_gate_calls+=1;assert cr is None
  else:
   candidate=out;cr=silu_and_mul(out,x);self.tileir_gate_calls+=1;assert cr is None
  # All inputs/outputs are contiguous and have no hidden logical gaps. Compare BF16 bits, not torch numeric equality.
  raw_equal=bool(torch.equal(out.view(torch.uint8),candidate.view(torch.uint8)));preserved=bool(torch.equal(x.view(torch.uint8),before.view(torch.uint8)))
  finite=bool(torch.isfinite(out).all().item()) and bool(torch.isfinite(candidate).all().item())
  layout=out.shape==candidate.shape and out.stride()==candidate.stride() and out.storage_offset()==candidate.storage_offset() and out.dtype==candidate.dtype
  alias=not overlap(out,x) and not overlap(candidate,x);passed=raw_equal and preserved and finite and layout and alias
  row=dict(path=name,shape=list(x.shape),phase='prefill' if x.shape[0]==1696 else 'decode',phase_evidence='shape-derived; request-level layer/shape census enforced',mode=self.mode,raw_equal=raw_equal,input_preserved=preserved,input_finite=input_finite,output_finite=finite,output_layout_equal=layout,output_input_disjoint=alias,return_is_none=True,passed=passed)
  self.rows.append(row)
  if not passed and self.first_mismatch is None:
   bank=self.output/'first_mismatch';bank.mkdir();art={}
   for label,t in [('input_before',before),('input_after',x),('native_or_actual',out),('candidate',candidate)]:
    f=bank/(label+'.pt');torch.save(t.detach().cpu(),f);art[label]=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset())
   self.first_mismatch=dict(call_index=len(self.rows)-1,row=row,artifacts=art)
  if not passed or len(self.rows)%32==0:self.save()
  assert passed,'SiLU raw/finite/layout contract HOLD';return None
 def start(self,mode):
  assert mode in ('shadow','replace') and (mode!='replace' or self.shadow_pass)
  self.mode=mode;self.rows=[];self.native_gate_calls=self.tileir_gate_calls=0;return dict(mode=mode)
 def summary(self):return dict(mode=self.mode,rows=self.rows,supported_calls=len(self.rows),native_gate_calls=self.native_gate_calls,tileir_gate_calls=self.tileir_gate_calls,shadow_pass=self.shadow_pass,first_mismatch=self.first_mismatch,supported_vendor_fallback_calls=0)
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2))
 def finish(self):
  counts=Counter((r['path'],tuple(r['shape'])) for r in self.rows)
  assert counts==Counter({(name,(m,18432)):n for name in self.modules for m,n in [(1696,1),(32,15)]})
  assert len(self.rows)==512 and all(r['passed'] for r in self.rows) and self.tileir_gate_calls==512 and self.native_gate_calls==(512 if self.mode=='shadow' else 0)
  if self.mode=='shadow':self.shadow_pass=True
  self.save();return self.summary()
 def close(self):
  conflicts=[]
  for m,had,prior,new in reversed(self.bindings):
   if m.__dict__.get('op') is not new:conflicts.append(type(m).__name__);continue
   if had:m.op=prior
   else:delattr(m,'op')
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
