"""One-request native-copy shadow only. No global mutation or second-model rearm."""
import functools,inspect
from .api import Copy,check,metadata,SHAPE
class Adapter:
 def __init__(self,save=None):self.bindings=[];self.rows=[];self.observed=[];self.layers=set();self.seen=set();self.save=save or (lambda receipt:None);self.context=None;self.failed=False;self.pending=[]
 def fail(self):self.failed=True
 def install(self,model):
  import torch
  from torch.utils._python_dispatch import TorchDispatchMode
  from vllm.forward_context import get_forward_context
  from gdn_post_conv_runtime_capture_v1.observer import phase_meta
  owner=self
  class Mode(TorchDispatchMode):
   def __torch_dispatch__(self,func,types,args=(),kwargs=None):return owner.observe(torch,func,args,kwargs or {})
  for name,m in model.named_modules():
   if not name.endswith('.linear_attn')or not hasattr(m,'chunk_gated_delta_rule'):continue
   if name in self.layers:raise RuntimeError('Duplicate GDN owner name')
   self.layers.add(name)
   orig=m._forward_core;sig=inspect.signature(orig);had='_forward_core'in vars(m);previous=vars(m).get('_forward_core')
   if 'core_attn_out' not in sig.parameters:raise RuntimeError('Core signature not preserved')
   @functools.wraps(orig)
   def wrap(*args,_m=m,_orig=orig,_sig=sig,_name=name,**kwargs):
    if owner.failed:raise RuntimeError('Copy observer failure latched')
    try:
     md=get_forward_context().attn_metadata[_m.prefix];phase=phase_meta(md)
     if phase['phase']!='prefill':return _orig(*args,**kwargs)
     if owner.context is not None or phase['speculative'] or phase['num_actual_tokens']!=1696 or len(owner.rows)>=24:raise RuntimeError('Unqualified copy phase')
     if _name in owner.seen:raise RuntimeError('Duplicate prefill owner')
     owner.seen.add(_name)
     b=_sig.bind(*args,**kwargs);b.apply_defaults();dst=b.arguments['core_attn_out']
     owner.context=dict(layer=_name,destination=dst,count=0)
     with Mode():ret=_orig(*args,**kwargs)
     if owner.context['count']!=1:raise RuntimeError('Expected exactly one owned output copy')
     return ret
    except BaseException:owner.fail();raise
    finally:owner.context=None
   setattr(m,'_forward_core',wrap);self.bindings.append((m,had,previous,wrap))
  if len(self.bindings)!=24:raise RuntimeError('Expected24 GDN bindings')
  return dict(bindings=24,scope='Native24 model copies plus private same-input TileIR shadow')
 def observe(self,t,func,args,kwargs):
  ctx=self.context
  if str(func)!='aten.copy_.default' or not ctx:return func(*args,**kwargs)
  dst,src=args[:2];target=ctx['destination']
  if not isinstance(dst,t.Tensor) or not isinstance(src,t.Tensor):self.failed=True;raise TypeError('Expected actual Tensor operands')
  if dst.untyped_storage().data_ptr()!=target.untyped_storage().data_ptr():return func(*args,**kwargs)
  try:
   def public_meta(x):return {k:v for k,v in metadata(x).items()if k not in ('ptr','storage_ptr')}
   self.observed.append(dict(layer=ctx['layer'],source=public_meta(src),destination=public_meta(dst)))
   self.save(dict(observed=self.observed,rows=self.rows,failed=self.failed))
   if ctx['count']!=0 or dst.data_ptr()!=target.data_ptr() or tuple(dst.shape)!=SHAPE:raise RuntimeError('Unexpected destination copy')
   if len(args)>3 or (args[2]if len(args)>2 else kwargs.get('non_blocking',False)):raise RuntimeError('Unexpected copy options')
   check(dst,src,t);ctx['count']+=1
   def backing(x):return x.as_strided((x.untyped_storage().nbytes()//2,),(1,),0)
   def raw(x):return x.detach().view(t.uint8).cpu().numpy().tobytes()
   sb,db=backing(src),backing(dst);source_before=raw(sb);before=raw(db);stream=t.cuda.current_stream(dst.device).cuda_stream
   # Clone full destination BEFORE native writes; private target independently
   # poisoned to the bitwise complement of source, so no-op cannot pass.
   private=db.clone();pd=private.as_strided(dst.shape,dst.stride(),dst.storage_offset())
   pd.view(t.int16).copy_(t.bitwise_not(src.view(t.int16)))
   if raw(pd)==raw(src):raise RuntimeError('Poison did not differ')
   copier=Copy();self.pending.append(copier);returned=copier(pd,src)
   t.cuda.current_stream(dst.device).synchronize();copier.release_after_sync()
   ret=func(*args,**kwargs)
   after=raw(db);candidate=raw(private);lo=dst.storage_offset()*2;hi=lo+dst.numel()*2
   if returned is not pd or ret is not dst or candidate!=after or after[:lo]!=before[:lo]or after[hi:]!=before[hi:]or raw(sb)!=source_before or t.cuda.current_stream(dst.device).cuda_stream!=stream:raise RuntimeError('Copy raw/readonly/identity/stream mismatch')
   self.rows.append(dict(layer=ctx['layer'],source=public_meta(src),destination=public_meta(dst),native_calls=1,candidate_calls=1,raw_exact=True,outside_preserved=True,source_readonly=True,poison_independent=True));return ret
  except BaseException:
   self.failed=True
   try:self.save(dict(observed=self.observed,rows=self.rows,failed=True))
   except BaseException:pass
   raise
 def finish(self):
  if self.failed or self.context is not None or len(self.rows)!=24 or self.seen!=self.layers or len(self.layers)!=24 or {r['layer']for r in self.rows}!=self.layers:raise RuntimeError('Incomplete copy shadow')
  return dict(complete=True,rows=self.rows,calls=24,model_copies_native=True,actual_replacement=False)
 def close(self):
  errors=[]
  for m,had,previous,wrap in reversed(self.bindings):
   if vars(m).get('_forward_core')is not wrap:errors.append('Foreign core binding');continue
   if had:setattr(m,'_forward_core',previous)
   else:delattr(m,'_forward_core')
  if errors:self.failed=True;raise RuntimeError(str(errors))
  self.bindings=[]
