"""Public helper wrapper; actual native state never modified by private shadow."""
import functools,inspect,json,re,hashlib
from pathlib import Path
from collections import Counter
from norm_runtime_qualification_v2.run import descriptor,identity
ATTR='_gdn_decode_core_runtime_v3'
def remap(slots,capacity):
 assert len(slots)==32 and len(set(slots))==32 and all(type(s)is int and 0<s<capacity for s in slots),'observed positive unique32-slot scope'
 selected=sorted(slots);lookup={s:i+1 for i,s in enumerate(selected)}
 return selected,[lookup[s]for s in slots]
def gpu_metrics(a,b,torch):
 assert a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape)
 bits=torch.int32 if a.dtype==torch.float32 else torch.int16
 aa=a.contiguous();bb=b.contiguous();rawbad=aa.view(bits)!=bb.view(bits)
 af=aa.float();bf=bb.float();fa=torch.isfinite(af);fb=torch.isfinite(bf);finite=fa&fb;err=(af-bf).abs();strict=finite&(err<=1e-5+1e-5*bf.abs())
 return dict(elements=a.numel(),raw_equal=bool((~rawbad).all().item()),raw_mismatches=int(rawbad.sum().item()),strict_pass=bool(strict.all().item()),strict_failures=int((~strict).sum().item()),candidate_nonfinite=int((~fa).sum().item()),reference_nonfinite=int((~fb).sum().item()),max_abs_finite=float(torch.where(finite,err,0).max().item()))
def raw_equal(a,b,torch):return a.dtype==b.dtype and tuple(a.shape)==tuple(b.shape) and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def numerical_pass(c):return all(x['raw_equal']and x['strict_pass']and x['candidate_nonfinite']==x['reference_nonfinite']==0 for x in c.values())
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.owners={};self.binding=None;self.rows=[];self.counts=Counter();self.mode='shadow';self.shadow_pass=False;self.scratch=None;self.first_mismatch=None
 def install(self,model):
  import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q
  for n,m in model.named_modules():
   if n.endswith('.linear_attn')and hasattr(m,'A_log')and hasattr(m,'dt_bias'):self.owners[(m.A_log.data_ptr(),m.dt_bias.data_ptr())]=n
  assert len(self.owners)==24 and {int(re.search(r'layers\.(\d+)\.linear_attn$',n)[1])for n in self.owners.values()}=={i for i in range(32)if i%4!=3}
  self.untouched={k:getattr(q,k)for k in ('causal_conv1d_fn','causal_conv1d_update','fused_post_conv_prep')};self.q=q
  key='fused_recurrent_gated_delta_rule_packed_decode';orig=getattr(q,key);sig=inspect.signature(orig)
  @functools.wraps(orig)
  def wrapper(*args,**kwargs):
   import torch
   from tileir_gdn_decode_core_v3 import packed_decode
   bound=sig.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments
   name=self.owners.get((v['A_log'].data_ptr(),v['dt_bias'].data_ptr()));assert name is not None
   self.counts[name]+=1
   state=v['initial_state'];out=v['out'];slots=v['ssm_state_indices'].detach().cpu().tolist();selected,mapped=remap(slots,state.shape[0]);assert tuple(state.shape[1:])==(32,128,128)and tuple(state.stride()[1:])==(16384,128,1)
   assert v['scale']==128**-0.5 and v['use_qk_l2norm_in_kernel']is True
   original_meta={k:descriptor(t)for k,t in v.items()if isinstance(t,torch.Tensor)}
   names=('mixed_qkv','a','b','A_log','dt_bias','ssm_state_indices')
   if self.mode=='replace':
    d={k:(t.detach()if isinstance(t,torch.Tensor)else t)for k,t in v.items()};got=packed_decode(**d)
    ok=isinstance(got,tuple)and len(got)==2 and got[0]is d['out']and got[1]is d['initial_state']and bool(torch.isfinite(out).all().item())
    statefinite=bool(torch.isfinite(state.index_select(0,torch.tensor(selected,dtype=torch.long,device=state.device))).all().item())
    ok=ok and statefinite
    assert ok,'production tuple/output/state finite HOLD'
    self.rows.append(dict(path=name,mode='replace',original_calls=0,tileir_calls=1,passed=ok,original_metadata=original_meta,tuple_identity=True,selected_state_finite=statefinite,scope='actual original state stride/offset and indices; no compact remapping'))
    return out,state # original caller object identities, not detached aliases
   assert self.mode=='shadow'
   if self.scratch is None:self.scratch=torch.empty((33,32,128,128),dtype=torch.float32,device=state.device)
   st=self.scratch;assert st.device==state.device;st[0].fill_(-7)
   ix=torch.tensor(selected,dtype=torch.long,device=state.device)
   pre=state.index_select(0,ix).detach();st[1:].copy_(pre)
   copies={}
   for k in names:
    t=v[k];buf=torch.empty_strided(t.shape,t.stride(),dtype=t.dtype,device=t.device);buf.copy_(t.detach());copies[k]=buf
   before={k:t.clone()for k,t in copies.items()}
   mapped_i=torch.tensor(mapped,dtype=v['ssm_state_indices'].dtype,device=state.device);mapped_before=mapped_i.clone()
   candidate_out=torch.empty_like(out)
   native=orig(*args,**kwargs) # exactly once on native trajectory
   expected=state.index_select(0,ix).detach()
   got=packed_decode(copies['mixed_qkv'],copies['a'],copies['b'],copies['A_log'],copies['dt_bias'],v['scale'],st,candidate_out,mapped_i,True)
   comparisons=dict(output=gpu_metrics(candidate_out,out,torch),state=gpu_metrics(st[1:],expected,torch))
   tupleok=isinstance(native,tuple)and len(native)==2 and native[0]is out and native[1]is state and got[0]is candidate_out and got[1]is st
   preserved=all(raw_equal(v[k],before[k],torch)and raw_equal(copies[k],before[k],torch)for k in names)and raw_equal(mapped_i,mapped_before,torch)
   state_preserved=raw_equal(state.index_select(0,ix),expected,torch);scratch_null=bool((st[0]==-7).all().item())
   row=dict(path=name,mode='shadow',original_calls=1,tileir_calls=1,comparisons=comparisons,tuple_identity=tupleok,readonly_inputs_preserved=preserved,native_selected_state_preserved_after_shadow=state_preserved,scratch_null_preserved=scratch_null,original_metadata=original_meta,scratch_state=descriptor(st),selected_slots=selected,remapped_slots=mapped,native_helper=identity(orig),scope='actual full selected values, compact state stride+slot remapping; not original state pitch comparison or full owner coverage')
   row['passed']=tupleok and preserved and state_preserved and scratch_null and numerical_pass(comparisons)
   if not row['passed']and self.first_mismatch is None:
    bank=self.output/'first_mismatch';bank.mkdir();arts={}
    tensors=dict(before);tensors.update(state_before=pre,native_state=expected,tile_state=st[1:],native_output=out,tile_output=candidate_out)
    for k,t in tensors.items():
     path=bank/(k+'.pt');torch.save(t.detach().cpu().contiguous(),path);arts[k]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),descriptor=descriptor(t))
    self.first_mismatch=dict(row=row,artifacts=arts)
   self.rows.append(row)
   if len(self.rows)%24==0 or not row['passed']:self.save()
   assert row['passed'],'First shadow numerical/contract HOLD; native call completed, no later launch'
   return native
  setattr(q,key,wrapper);self.binding=(q,key,orig,wrapper)
  return dict(modules=list(self.owners.values()),scope='decode core only360calls; prefill/convolution/postconv bindings untouched')
 def start(self,mode):
  assert mode in('shadow','replace')
  if mode=='replace':assert self.shadow_pass
  self.mode=mode;self.rows=[];self.counts.clear();return dict(mode=mode)
 def summary(self):return dict(mode=self.mode,rows=self.rows,counts=dict(self.counts),supported_calls=len(self.rows),supported_vendor_fallback_calls=0,shadow_pass=self.shadow_pass,first_mismatch=self.first_mismatch,other_bindings_unchanged=all(getattr(self.q,k)is f for k,f in self.untouched.items()))
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2)+'\n')
 def finish(self):
  assert set(self.counts)==set(self.owners.values())and set(self.counts.values())=={15}and len(self.rows)==360
  assert all(getattr(self.q,k)is f for k,f in self.untouched.items())
  if self.mode=='shadow':self.shadow_pass=all(r['passed']for r in self.rows)
  self.save();return self.summary()
 def close(self):
  if self.binding is not None:
   q,k,orig,wrap=self.binding;assert getattr(q,k)is wrap;setattr(q,k,orig);self.binding=None
  self.scratch=None;assert all(getattr(self.q,k)is f for k,f in self.untouched.items());return dict(restored=True)
def install(model,output):
 assert not hasattr(model,ATTR);a=Adapter(output);r=a.install(model);setattr(model,ATTR,a);return r
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return dict(restored=True)
 r=a.close();delattr(model,ATTR);return r
