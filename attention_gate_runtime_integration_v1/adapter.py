"""Scoped public forward copy: only sigmoid-gate expression is replaced."""
import types,json,re,hashlib
from collections import Counter
from pathlib import Path
from norm_runtime_qualification_v2.run import logical,descriptor,aliases,identity
from norm_runtime_qualification_v2.compare import metrics
ATTR='_tileir_attention_gate_integration_v1'
SOURCE_SHA='09ea2a385259edc9feadc79e304e86a31582b9770f80b8003d2d641782405818'
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.saved=[];self.installed=[];self.mode='shadow';self.rows=[];self.shadow_pass=False;self.first_mismatch=None;self.native_gate_calls=0;self.tileir_gate_calls=0
 def install(self,model):
  found=[(n,m) for n,m in model.named_modules() if re.search(r'(?:^|\.)layers\.(3|7|11|15|19|23|27|31)\.self_attn$',n)]
  assert len(found)==8 and {int(re.search(r'layers\.(\d+)',n)[1]) for n,m in found}==set(range(3,32,4))
  try:
   for name,m in found:
    orig=m.forward;ident=identity(orig)
    assert ident['qualname']=='Qwen3NextAttention.forward' and ident['source_sha256']==SOURCE_SHA and m.attn_output_gate is True
    self.installed.append(dict(path=name,forward=ident,project=identity(m._project_qkv_gate),attention=identity(m.attn.forward),consumer=identity(m.o_proj.forward)))
    had='forward' in m.__dict__;prior=m.__dict__.get('forward')
    def forward(this,positions,hidden_states,_name=name):
     # Public pinned forward ordering and tuple conventions retained verbatim.
     qkv,_=this.qkv_proj(hidden_states)
     q,k,v,gate=this._project_qkv_gate(qkv,positions)
     attn_output=this.attn(q,k,v)
     assert gate is not None,'Output-gate contract changed'
     attn_output=self.apply_gate(_name,attn_output,gate)
     output,_=this.o_proj(attn_output)
     return output
    bound=types.MethodType(forward,m);self.saved.append((m,had,prior,bound));m.forward=bound
   return dict(modules=self.installed,eligible_modules=8,scope='Eager pinned public forward copy; gate only')
  except BaseException:self.close();raise
 def apply_gate(self,name,x,g):
  import torch
  from tileir_vllm_attention_gate_v1 import attention_sigmoid_gate
  if self.mode=='replace':
   assert self.shadow_pass
   out=attention_sigmoid_gate(x.detach(),g.detach());self.tileir_gate_calls+=1
   self.rows.append(dict(path=name,shape=list(x.shape),mode='replace'));return out
  assert self.mode=='shadow'
  vals=dict(x=x,gate=g);before={k:logical(t,torch) for k,t in vals.items()}
  native=x*torch.sigmoid(g);self.native_gate_calls+=1
  assert before=={k:logical(t,torch) for k,t in vals.items()},'Native mutated inputs'
  tile=attention_sigmoid_gate(x.detach(),g.detach());self.tileir_gate_calls+=1
  comp=metrics(tile,native);nd=descriptor(native);td=descriptor(tile)
  row=dict(path=name,shape=list(x.shape),mode='shadow',comparison=comp,input_descriptors={k:descriptor(t) for k,t in vals.items()},inputs_preserved=before=={k:logical(t,torch) for k,t in vals.items()},alias_equal=aliases(dict(vals,output=native))==aliases(dict(vals,output=tile)),layout_equal=all(nd[k]==td[k] for k in ('shape','stride','storage_offset','dtype')))
  row['passed']=row['inputs_preserved'] and row['alias_equal'] and row['layout_equal'] and comp['raw_equal'] and comp['strict_pass'] and comp['candidate_nonfinite']==comp['reference_nonfinite']==0
  if not row['passed'] and self.first_mismatch is None:
   bank=self.output/'first_mismatch';bank.mkdir(exist_ok=False);art={}
   for k,t in dict(vals,native=native,candidate=tile).items():
    f=bank/(k+'.pt');torch.save(t.detach().cpu().contiguous(),f);art[k]=dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),descriptor=descriptor(t))
   self.first_mismatch=dict(call_index=len(self.rows),row=row,artifacts=art)
  self.rows.append(row)
  if not row['passed'] or len(self.rows)%8==0:self.save()
  return native
 def start(self,mode):
  assert mode in ('shadow','replace')
  if mode=='replace':assert self.shadow_pass
  self.mode=mode;self.rows=[];self.native_gate_calls=0;self.tileir_gate_calls=0;return dict(mode=mode)
 def summary(self):return dict(mode=self.mode,rows=self.rows,supported_calls=len(self.rows),native_gate_calls=self.native_gate_calls,tileir_gate_calls=self.tileir_gate_calls,shadow_pass=self.shadow_pass,first_mismatch=self.first_mismatch,supported_vendor_fallback_calls=0)
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2)+'\n')
 def finish(self):
  counts=Counter((x['path'],tuple(x['shape'])) for x in self.rows)
  assert counts==Counter({(n['path'],(m,4096)):k for n in self.installed for m,k in [(1696,1),(32,15)]})
  assert self.tileir_gate_calls==128 and self.native_gate_calls==(128 if self.mode=='shadow' else 0)
  if self.mode=='shadow':self.shadow_pass=all(x['passed'] for x in self.rows)
  self.save();return self.summary()
 def close(self):
  conflicts=[]
  for m,had,prior,bound in reversed(self.saved):
   if m.__dict__.get('forward') is not bound:conflicts.append(type(m).__name__);continue
   if had:m.forward=prior
   else:delattr(m,'forward')
  self.saved=[];assert not conflicts,conflicts;return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR);a=Adapter(output);r=a.install(model);setattr(model,ATTR,a);return r
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 r=a.close();delattr(model,ATTR);return r
