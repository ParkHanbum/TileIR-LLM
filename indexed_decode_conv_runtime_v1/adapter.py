"""Native state never modified by shadow candidate; indexed scratch preserves strides."""
import inspect,functools,json,re,hashlib
from pathlib import Path
from collections import Counter
from norm_runtime_qualification_v2.run import logical,descriptor,identity
from norm_runtime_qualification_v2.compare import metrics
ATTR='_indexed_decode_conv_runtime_v1'
def remap_slots(slots,capacity):
 assert len(slots)==32 and all(type(s)is int and 0<=s<capacity for s in slots)
 selected=sorted(set(s for s in slots if s!=0));assert len(selected)==sum(s!=0 for s in slots)
 lookup={s:i+1 for i,s in enumerate(selected)}
 return selected,[0 if s==0 else lookup[s] for s in slots]
def extent(shape,stride):return 1+sum((n-1)*s for n,s in zip(shape,stride))
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.bindings=[];self.weights={};self.mode='shadow';self.rows=[];self.counts=Counter();self.shadow_pass=False;self.first_mismatch=None;self.scratch=None
 def install(self,model):
  import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q
  for n,m in model.named_modules():
   if n.endswith('.linear_attn') and hasattr(m,'conv1d'):
    assert tuple(m.conv1d.weight.shape)==(8192,1,4);self.weights[m.conv1d.weight.data_ptr()]=n
  assert len(self.weights)==24
  assert {int(re.search(r'layers\.(\d+)\.linear_attn$',n)[1]) for n in self.weights.values()}=={i for i in range(32) if i%4!=3}
  try:
   for api,phase in [('causal_conv1d_fn','prefill'),('causal_conv1d_update','decode')]:
    orig=getattr(q,api);sig=inspect.signature(orig)
    @functools.wraps(orig)
    def wrapper(*args,_orig=orig,_sig=sig,_phase=phase,**kwargs):
     v=_sig.bind(*args,**kwargs);v.apply_defaults();v=v.arguments
     name=self.weights.get(v['weight'].data_ptr());assert name is not None
     self.counts[name+':'+_phase]+=1
     if _phase=='prefill':return _orig(*args,**kwargs)
     import torch
     from tileir_indexed_decode_conv_v1 import causal_conv1d_update_indexed
     x,state,w,idx=v['x'],v['conv_state'],v['weight'],v['conv_state_indices']
     assert v['bias'] is None and v['out'] is None and v['activation']=='silu' and v['null_block_id']==0
     assert all(v[k] is None for k in ('num_accepted_tokens','query_start_loc','block_idx_last_scheduled_token','initial_state_idx')) and v['max_query_len']==-1
     assert tuple(x.shape)==(32,8192) and tuple(state.shape[1:])==(8192,3)
     slots=idx.detach().cpu().tolist();selected,mapped=remap_slots(slots,state.shape[0])
     if self.mode=='replace':
      out=causal_conv1d_update_indexed(x.detach(),state.detach(),w.detach(),idx.detach())
      alias=out is not x and out.untyped_storage().data_ptr()==x.untyped_storage().data_ptr() and tuple(out.stride())==tuple(x.stride()) and out.storage_offset()==x.storage_offset()
      assert alias,'Replacement output alias/layout mismatch'
      self.rows.append(dict(path=name,mode='replace',shape=list(x.shape),original_calls=0,tileir_calls=1,passed=alias,original_x=descriptor(x),original_state=descriptor(state),output=descriptor(out),output_alias_pass=alias));return out
     assert self.mode=='shadow'
     shape=(33,8192,3);ss=tuple(state.stride());xs=tuple(x.stride());signature=(str(x.device),ss,xs)
     if self.scratch is None:
      assert extent(shape,ss)*2<80*1024*1024
      sb=torch.empty(extent(shape,ss),device=x.device,dtype=state.dtype);sxbuf=torch.empty(extent(x.shape,xs),device=x.device,dtype=x.dtype)
      self.scratch=(signature,torch.as_strided(sb,shape,ss),torch.as_strided(sxbuf,x.shape,xs))
     key,st,sx=self.scratch;assert key==signature
     pre_state=state[selected].detach().clone();pre_x=x.detach().clone();st[1:len(selected)+1].copy_(pre_state);sx.copy_(x)
     mi=torch.tensor(mapped,dtype=idx.dtype,device=idx.device)
     wbefore=logical(w,torch);ibefore=logical(idx,torch)
     native=_orig(*args,**kwargs) # exactly once, real native trajectory mutation
     native_post=state[selected].detach().clone();native_x_hash=logical(x,torch);native_y_hash=logical(native,torch)
     tile=causal_conv1d_update_indexed(sx,st,w.detach(),mi)
     comparisons=dict(output=metrics(tile,native),state=metrics(st[1:len(selected)+1],native_post))
     alias=(native is not x and native.untyped_storage().data_ptr()==x.untyped_storage().data_ptr() and tile is not sx and tile.untyped_storage().data_ptr()==sx.untyped_storage().data_ptr())
     row=dict(path=name,mode='shadow',shape=list(x.shape),comparisons=comparisons,original_calls=1,tileir_calls=1,native_helper=identity(_orig),state_shape=list(state.shape),state_stride=list(ss),scratch_shape=list(shape),scratch_stride=list(st.stride()),selected_slots=selected,remapped_slots=mapped,native_alias_pass=alias,layout_pass=tuple(native.stride())==tuple(tile.stride())==xs,weight_indices_preserved=logical(w,torch)==wbefore and logical(idx,torch)==ibefore,native_state_preserved_after_shadow=logical(state[selected],torch)==logical(native_post,torch),native_x_preserved_after_shadow=logical(x,torch)==native_x_hash and logical(native,torch)==native_y_hash,original_x=descriptor(x),original_state=descriptor(state),scratch_x=descriptor(sx))
     row['passed']=all(row[k] for k in ('native_alias_pass','layout_pass','weight_indices_preserved','native_state_preserved_after_shadow','native_x_preserved_after_shadow')) and all(c['raw_equal'] and c['strict_pass'] and c['candidate_nonfinite']==c['reference_nonfinite']==0 for c in comparisons.values())
     if not row['passed'] and self.first_mismatch is None:
      bank=self.output/'first_mismatch';bank.mkdir();artifacts={}
      for k,t in dict(x=pre_x,state=pre_state,w=w,indices=idx,nativeY=native,tileY=tile,nativeState=native_post,tileState=st[1:len(selected)+1]).items():
       path=bank/(k+'.pt');torch.save(t.detach().cpu().contiguous(),path);artifacts[k]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),descriptor=descriptor(t))
      self.first_mismatch=dict(row=row,artifacts=artifacts)
     self.rows.append(row)
     if len(self.rows)%24==0 or not row['passed']:self.save()
     return native
    setattr(q,api,wrapper);self.bindings.append((q,api,orig,wrapper))
   return dict(modules=list(self.weights.values()),decode_scope='360 selected calls; prefill24 native unchanged')
  except BaseException:self.close();raise
 def start(self,mode):
  assert mode in ('shadow','replace')
  if mode=='replace':assert self.shadow_pass
  self.mode=mode;self.rows=[];self.counts.clear();return {'mode':mode}
 def summary(self):return dict(mode=self.mode,rows=self.rows,counts=dict(self.counts),supported_calls=len(self.rows),supported_vendor_fallback_calls=0,shadow_pass=self.shadow_pass,first_mismatch=self.first_mismatch,scope='selected state scratch; original strides retained, slots remapped; wholeowner storage not compared')
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2)+'\n')
 def finish(self):
  assert set(self.counts)=={n+':'+p for n in self.weights.values() for p in ('prefill','decode')}
  assert all(v==(1 if k.endswith(':prefill') else 15) for k,v in self.counts.items()) and len(self.rows)==360
  if self.mode=='shadow':self.shadow_pass=all(row['passed'] for row in self.rows)
  self.save();return self.summary()
 def close(self):
  conflicts=[]
  for q,k,orig,wrap in reversed(self.bindings):
   if getattr(q,k)is not wrap:conflicts.append(k)
   else:setattr(q,k,orig)
  self.bindings=[];self.scratch=None;assert not conflicts,conflicts;return {'restored':True}
def install(model,output):
 assert not hasattr(model,ATTR);a=Adapter(output);r=a.install(model);setattr(model,ATTR,a);return r
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return {'restored':True}
 r=a.close();delattr(model,ATTR);return r
