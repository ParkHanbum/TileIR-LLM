"""Lean public sampler boundaries. Numerical APIs and scope guards are unchanged.
No snapshots, comparisons, per-call logs, or explicit synchronization here.
"""
import inspect,json
from pathlib import Path
from .support import Base,phase_index
from block_gather_runtime_capture_v1.observer import phase_metadata
from norm_runtime_contract_capture_v1.observer import identity
from tp1_embedding_runtime_capture_v2.observer import meta

class _Sampler(Base):
 helper_name=None
 def install_classes(self,S,module):
  original=S.__call__;helper=getattr(module,self.helper_name);sig=inspect.signature(helper)
  def call(owner,logits,batch,*args,**kwargs):
   self.guard()
   try:
    assert self.pending is None and owner.logprobs_mode=='raw_logprobs' and owner.trace_replay_state is None
    metadata=phase_metadata(batch)
    assert metadata['num_reqs']==32 and metadata['phase']==phase_index(sum(self.counts.values()))
    ctx=dict(owner=owner,batch=batch,metadata=metadata,calls=0);self.pending=ctx
    result=original(owner,logits,batch,*args,**kwargs)
    assert ctx['calls']==1,'Expected exactly one selected helper per sampler call'
    return result
   except BaseException as e:self.controller.latch(self.key,e);raise
   finally:self.pending=None
  def replacement(*args,**kwargs):
   self.guard()
   try:
    ctx=self.pending;assert ctx is not None,'Selected helper outside sampler context'
    ctx['calls']+=1;assert ctx['calls']==1
    bound=sig.bind(*args,**kwargs);bound.apply_defaults()
    return self.apply(bound.arguments,ctx)
   except BaseException as e:self.controller.latch(self.key,e);raise
  self.bind(S,'__call__',call);self.bind(module,self.helper_name,replacement)
 def finish(self):
  result=super().finish()
  assert self.counts[('sampler','prefill')]==1 and self.counts[('sampler','decode')]==15
  return result

class Counts(_Sampler):
 helper_name='get_num_sampled_and_rejected'
 def install(self,model):
  from vllm.v1.worker.gpu.sample import sampler
  from tileir_vllm_sample_counts_v1.api import get_num_sampled_and_rejected
  self.candidate=get_num_sampled_and_rejected
  self.install_classes(sampler.Sampler,sampler)
  return dict(boundary=self.helper_name,scope='N32 no-spec actual UVA prefill; same sampled object returned')
 def apply(self,v,ctx):
  # Parameter names come from the pinned public helper. Preserve all five actual views.
  sampled,seq,prefix,idx,prefill=(v[k]for k in ('num_sampled','seq_lens','cu_num_logits','idx_mapping','prefill_len'))
  b=ctx['batch'];assert seq is b.seq_lens and prefix is b.cu_num_logits and idx is b.idx_mapping
  assert tuple(sampled.shape)==(32,)
  owner=ctx['owner'].req_states.prefill_len;active=owner.pool._uva_bufs[owner.pool._curr]
  assert prefill is owner.gpu and type(active).__name__=='UvaBuffer' and active.cpu.is_pinned()
  assert prefill.data_ptr()==active.cpu.data_ptr()==active._uva.data_ptr()
  # Keep every involved pinned owner alive in the enclosing controller. No sync added.
  if not hasattr(self.controller,'sampler_owner_refs'):self.controller.sampler_owner_refs=[]
  refs=self.controller.sampler_owner_refs
  if not any(x[0]is active for x in refs):refs.append((active,owner,owner.pool,active.cpu,active._uva,prefill))
  out=self.call('sampler',ctx['metadata']['phase'],self.candidate,sampled,seq,prefix,idx,prefill)
  assert isinstance(out,tuple)and len(out)==2 and out[0]is sampled
  return out

class Greedy(_Sampler):
 helper_name='gumbel_sample'
 def install(self,model):
  from vllm.v1.worker.gpu.sample import sampler
  from tileir_vllm_greedy_sample_v1 import greedy_sample
  assert identity(sampler.gumbel_sample)['source_sha256']=='c14a5fa1899a9ff320c62f29bd161be4a4f0fa124ecf26a28681f0f3d143868e'
  self.candidate=greedy_sample;self.install_classes(sampler.Sampler,sampler)
  return dict(boundary=self.helper_name,scope='temperature-zero only; original API internal finite guard remains')
 def apply(self,v,ctx):
  import torch
  from tileir_vllm_greedy_sample_v1.api import validate_descriptor
  assert v['apply_temperature']is False and v['is_drafting']is False and v['use_fp64']is False
  assert all(v[k]is None for k in ('logits_cache','logits_cache_col','logits_cache_source'))
  x,idx,temp=v['logits'],v['expanded_idx_mapping'],v['temperature'];validate_descriptor(meta(x))
  # These reads enforce supported greedy scope; removing them would broaden support.
  assert idx.dtype==torch.int64 and tuple(idx.shape)==(32,)and bool(((idx>=0)&(idx<len(temp))).all())
  assert bool((temp.index_select(0,idx)==0).all())
  return self.call('sampler',ctx['metadata']['phase'],self.candidate,x)

class SamplerThree(_Sampler):
 helper_name='compute_topk_scores'
 def install(self,model):
  from vllm.v1.worker.gpu.sample import sampler,logprob
  from sampler_three_ordered_runtime_integration_v2.adapter import build_copy,ROOT
  from tileir_vllm_selected_ranks_v1 import selected_token_ranks
  from tileir_vllm_top5_v2 import top5_indices
  from tileir_vllm_selected_logprobs_v2 import compute_token_logprobs
  assert sampler.compute_topk_scores is logprob.compute_topk_scores
  spec=json.loads((ROOT/'SOURCE.json').read_text())
  assert identity(sampler.compute_topk_scores)['source_sha256']==spec['sha256']
  self.candidate=build_copy(logprob,selected_token_ranks,top5_indices,compute_token_logprobs)
  self.install_classes(sampler.Sampler,sampler)
  return dict(boundary=self.helper_name,copy_sha256=spec['copy_sha256'],scope='Same Top5/ranks/ordered-score APIs; cat/output construction retained')
 def apply(self,v,ctx):
  from tileir_vllm_selected_ranks_v1.api import validate_descriptors
  assert v['num_logprobs']==5 and v['max_per_req_token_ids']==0 and v['logits_mode']is False and v['cu_num_logits']is None
  validate_descriptors(meta(v['logits']),meta(v['sampled_token_ids']))
  return self.call('sampler',ctx['metadata']['phase'],self.candidate,**v)
 def finish(self):
  result=super().finish();result.update(logical_suboperation_calls=48,suboperation_counts=dict(top5=16,ranks=16,selected_logprobs=16),native_cat_retained=True);return result
