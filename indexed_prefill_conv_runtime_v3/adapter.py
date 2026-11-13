"""Public ordinary prefill only; shadow never exposes original state to candidate."""
import inspect,functools,json,re,hashlib
from pathlib import Path
from collections import Counter
from norm_runtime_qualification_v2.run import logical,descriptor,identity,aliases
ATTR='_indexed_prefill_conv_runtime_v3'
def remap_slots(slots,capacity):
 assert len(slots)==32 and all(type(s)is int and 0<s<capacity for s in slots)
 selected=sorted(slots);assert len(set(selected))==32
 lookup={s:i+1 for i,s in enumerate(selected)}
 return selected,[lookup[s] for s in slots]
def extent(shape,stride):return 1+sum((n-1)*s for n,s in zip(shape,stride))
def full_hash(t,torch):
 b=torch.empty(0,dtype=torch.uint8,device=t.device).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,))
 return logical(b,torch)
def detached_same_view(t):
 d=t.detach()
 assert d.data_ptr()==t.data_ptr() and d.untyped_storage().data_ptr()==t.untyped_storage().data_ptr() and descriptor(d)==descriptor(t) and not d.requires_grad
 return d
def fresh_output(out,inputs):
 if tuple(out.shape)!=(8192,1696) or tuple(out.stride())!=(1,8192) or out.storage_offset()!=0 or str(out.dtype)!='torch.bfloat16':return False
 return not any(v['same_object']or v['same_storage']or v['storage_overlap']for k,v in aliases(dict(inputs,output=out)).items()if'output'in k.split('|'))
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.bindings=[];self.weights={};self.mode='shadow';self.rows=[];self.counts=Counter();self.shadow_pass=False;self.first_mismatch=None;self.scratch=None
 def install(self,model):
  import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as q
  for n,m in model.named_modules():
   if n.endswith('.linear_attn') and hasattr(m,'conv1d'):
    assert tuple(m.conv1d.weight.shape)==(8192,1,4);key=m.conv1d.weight.data_ptr();assert key not in self.weights;self.weights[key]=n
  assert len(self.weights)==24 and {int(re.search(r'layers\.(\d+)\.linear_attn$',n)[1])for n in self.weights.values()}=={i for i in range(32)if i%4!=3}
  try:
   self.update_binding=(q,q.causal_conv1d_update)
   for api,phase in [('causal_conv1d_fn','prefill')]:
    orig=getattr(q,api);sig=inspect.signature(orig)
    @functools.wraps(orig)
    def wrapper(*args,_orig=orig,_sig=sig,_phase=phase,**kwargs):
     v=_sig.bind(*args,**kwargs);v.apply_defaults();v=v.arguments
     name=self.weights.get(v['weight'].data_ptr());assert name is not None;self.counts[name+':'+_phase]+=1
     import torch
     from tileir_indexed_prefill_conv_v1 import causal_conv1d_prefill_indexed
     from norm_runtime_qualification_v2.compare import metrics
     x,state,w,idx,qstart,flags=(v[k]for k in ('x','conv_states','weight','cache_indices','query_start_loc','has_initial_state'))
     assert v['bias']is None and v['activation']=='silu' and v['null_block_id']==0 and v['pad_slot_id']==-1 and v['block_size_to_align']==0
     assert all(v[k]is None for k in ('block_idx_first_scheduled_token','block_idx_last_scheduled_token','initial_state_idx','num_computed_tokens'))
     assert tuple(x.shape)==(8192,1696) and tuple(state.shape[1:])==(8192,3)
     starts=qstart.detach().cpu().tolist();assert starts==list(range(0,1697,53)) and not any(flags.detach().cpu().tolist())
     slots=idx.detach().cpu().tolist();selected,mapped=remap_slots(slots,state.shape[0]);ss=tuple(state.stride());xs=tuple(x.stride())
     original_inputs=dict(x=x,weight=w,indices=idx,query=qstart,flags=flags)
     before_hash={k:full_hash(t,torch)for k,t in original_inputs.items()};wd=detached_same_view(w)
     row=dict(path=name,mode=self.mode,shape=list(x.shape),original_x=descriptor(x),original_state=descriptor(state),original_weight_requires_grad=w.requires_grad,candidate_weight_same_view=True,inputs_finite={'x':bool(torch.isfinite(x).all()),'weight':bool(torch.isfinite(w).all())})
     # Raw expected state is final3 original X tokens for each selected slot.
     expected_state=torch.stack([x[:,starts[slots.index(slot)+1]-3:starts[slots.index(slot)+1]].detach()for slot in selected])
     if self.mode=='replace':
      out=causal_conv1d_prefill_indexed(detached_same_view(x),wd,detached_same_view(state),qstart,idx,flags)
      row.update(original_calls=0,tileir_calls=1,output=descriptor(out),output_alias_pass=fresh_output(out,dict(original_inputs,state=state)),comparisons={'state':metrics(state[selected],expected_state)})
      row['inputs_fullstorage_preserved']={k:full_hash(t,torch)==before_hash[k]for k,t in original_inputs.items()}
      row['output_finite']=bool(torch.isfinite(out).all())
      row['passed']=row['output_finite']and all(row['inputs_finite'].values())and row['output_alias_pass']and all(row['inputs_fullstorage_preserved'].values())and all(c['raw_equal']and c['strict_pass']and c['candidate_nonfinite']==c['reference_nonfinite']==0 for c in row['comparisons'].values())
      if not row['passed'] and self.first_mismatch is None:
       bank=self.output/'first_mismatch';bank.mkdir();artifacts={}
       for k,t in dict(x=x,w=w,indices=idx,query=qstart,initial=flags,tileY=out,tileState=state[selected],expectedState=expected_state).items():
        path=bank/(k+'.pt');torch.save(t.detach().cpu().contiguous(),path);artifacts[k]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),descriptor=descriptor(t))
       self.first_mismatch=dict(row=row,artifacts=artifacts)
      self.rows.append(row);self.save();assert row['passed'],'Replacement input/state/alias HOLD'
      return out
     assert self.mode=='shadow'
     shape=(33,8192,3);signature=(str(x.device),ss,xs)
     if self.scratch is None:
      assert extent(shape,ss)*2<80*1024*1024 and extent(x.shape,xs)*2<64*1024*1024
      sb=torch.empty(extent(shape,ss),device=x.device,dtype=state.dtype);xb=torch.empty(extent(x.shape,xs),device=x.device,dtype=x.dtype)
      self.scratch=(signature,torch.as_strided(sb,shape,ss),torch.as_strided(xb,x.shape,xs))
     key,st,sx=self.scratch;assert key==signature
     pre_state=state[selected].detach().clone();st[1:33].copy_(pre_state);sx.copy_(x);mi=torch.tensor(mapped,dtype=idx.dtype,device=idx.device)
     native=_orig(*args,**kwargs) # exact original args and weight identity, exactly once
     native_post=state[selected].detach().clone();native_hash=logical(native,torch)
     tile=causal_conv1d_prefill_indexed(sx,wd,st,qstart,mi,flags)
     row.update(original_calls=1,tileir_calls=1,native_helper=identity(_orig),comparisons={'output':metrics(tile,native),'state':metrics(st[1:33],native_post),'native_state_semantics':metrics(native_post,expected_state)},scratch_shape=list(shape),scratch_stride=list(st.stride()),scratch_x=descriptor(sx),selected_slots=selected,remapped_slots=mapped)
     row['inputs_fullstorage_preserved']={k:full_hash(t,torch)==before_hash[k]for k,t in original_inputs.items()}
     row['scratch_x_unchanged']=logical(sx,torch)==logical(x,torch)
     row['native_post_unchanged']=logical(state[selected],torch)==logical(native_post,torch)and logical(native,torch)==native_hash
     row['native_output_alias_pass']=fresh_output(native,dict(original_inputs,state=state))
     row['tile_output_alias_pass']=fresh_output(tile,dict(original_inputs,state=st,scratch_x=sx,remapped_indices=mi))
     row['output_finite']=row['comparisons']['output']['candidate_nonfinite']==row['comparisons']['output']['reference_nonfinite']==0
     row['passed']=row['output_finite']and all(row['inputs_finite'].values())and all(row[k]for k in ('scratch_x_unchanged','native_post_unchanged','native_output_alias_pass','tile_output_alias_pass'))and all(row['inputs_fullstorage_preserved'].values())and all(c['raw_equal']and c['strict_pass']and c['candidate_nonfinite']==c['reference_nonfinite']==0 for c in row['comparisons'].values())
     if not row['passed']and self.first_mismatch is None:
      bank=self.output/'first_mismatch';bank.mkdir();artifacts={}
      for k,t in dict(x=x,state=pre_state,w=w,indices=idx,query=qstart,initial=flags,nativeY=native,tileY=tile,nativeState=native_post,tileState=st[1:33]).items():
       path=bank/(k+'.pt');torch.save(t.detach().cpu().contiguous(),path);artifacts[k]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),descriptor=descriptor(t))
      self.first_mismatch=dict(row=row,artifacts=artifacts)
     self.rows.append(row);self.save();assert row['passed'],'Prefill shadow first mismatch; replacement forbidden'
     return native
    setattr(q,api,wrapper);self.bindings.append((q,api,orig,wrapper))
   return dict(modules=list(self.weights.values()),prefill_scope='24 selected fn calls only; this adapter never wraps update or GEMM',update_helper=identity(self.update_binding[1]))
  except BaseException:self.close();raise
 def start(self,mode):
  assert mode in ('shadow','replace')
  if mode=='replace':assert self.shadow_pass
  self.mode=mode;self.rows=[];self.counts.clear();return {'mode':mode}
 def summary(self):return dict(mode=self.mode,rows=self.rows,counts=dict(self.counts),supported_calls=len(self.rows),supported_vendor_fallback_calls=0,shadow_pass=self.shadow_pass,first_mismatch=self.first_mismatch,update_binding_is_install_time_object=self.update_binding[0].causal_conv1d_update is self.update_binding[1],scope='Prefill selected-state scratch: original pitches retained, slots remapped; no original fullowner comparison or direct update-call census')
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2)+'\n')
 def finish(self):
  assert set(self.counts)=={n+':prefill'for n in self.weights.values()}
  assert all(v==1 for v in self.counts.values())and len(self.rows)==24
  if self.mode=='shadow':self.shadow_pass=all(row['passed']for row in self.rows)
  self.save();return self.summary()
 def close(self):
  conflicts=[]
  for q,k,orig,wrap in reversed(self.bindings):
   if getattr(q,k)is not wrap:conflicts.append(k)
   else:setattr(q,k,orig)
  self.bindings=[];self.scratch=None;assert not conflicts,conflicts;return {'restored':True,'update_binding_is_install_time_object':self.update_binding[0].causal_conv1d_update is self.update_binding[1]}
def install(model,output):
 assert not hasattr(model,ATTR);a=Adapter(output);r=a.install(model);setattr(model,ATTR,a);return r
def start(model,mode):return getattr(model,ATTR).start(mode)
def finish(model):return getattr(model,ATTR).finish()
def close(model):
 a=getattr(model,ATTR,None)
 if a is None:return {'restored':True}
 r=a.close();delattr(model,ATTR);return r
