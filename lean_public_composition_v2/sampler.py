"""New lean sampler boundaries; existing public numeric APIs/guards unchanged."""
import functools,hashlib,inspect,json,textwrap
from pathlib import Path
from lean_public_composition_v1.sampler import SamplerThree,_Sampler
from lean_public_composition_v1.support import Base,phase_index
from norm_runtime_contract_capture_v1.observer import identity

class SamplerFour(SamplerThree):
 def install(self,model):
  from vllm.v1.worker.gpu.sample import sampler,logprob
  from sampler_four_runtime_integration_v1.adapter import build_copy,ROOT
  from tileir_vllm_selected_ranks_v1 import selected_token_ranks
  from tileir_vllm_top5_v2 import top5_indices
  from tileir_vllm_selected_logprobs_v2 import compute_token_logprobs
  from tileir_vllm_sampler_cat_v1 import concat_sampled_top5
  assert sampler.compute_topk_scores is logprob.compute_topk_scores
  spec=json.loads((ROOT/'SOURCE.json').read_text())
  assert identity(sampler.compute_topk_scores)['source_sha256']==spec['sha256']
  self.cat_calls=0;self.cat_api=concat_sampled_top5
  self.candidate=build_copy(logprob,selected_token_ranks,top5_indices,compute_token_logprobs,self.cat)
  self.install_classes(sampler.Sampler,sampler)
  return dict(boundary=self.helper_name,copy_sha256=spec['copy_sha256'],scope='Same three APIs plus unchanged public cat API; output construction retained')
 def start(self):super().start();self.cat_calls=0
 def cat(self,chosen,top5):
  self.guard();assert self.pending is not None and self.cat_calls==sum(self.counts.values())
  result=self.cat_api(chosen,top5);self.cat_calls+=1;return result
 def finish(self):
  r=_Sampler.finish(self);assert self.cat_calls==16
  r.update(logical_suboperation_calls=64,suboperation_counts=dict(top5=16,ranks=16,selected_logprobs=16,cat=16),native_cat_retained=False);return r

class SamplerFP32(Base):
 def install(self,model):
  from vllm.v1.worker.gpu.sample.sampler import Sampler
  from tileir_vllm_sampler_fp32_copy_v1 import sampler_fp32_copy
  from sampler_fp32_copy_runtime_integration_v1.adapter import R
  self.api=sampler_fp32_copy;self.current=None;self.sampler_owner=None
  old=Sampler.apply_sampling_params;pin=json.loads((R/'SOURCE.json').read_text())
  assert hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest()==pin['sha256']
  assert textwrap.dedent(inspect.getsource(old)).strip()==(R/'method.original.py.txt').read_text().strip()
  ns=dict(old.__globals__);ns['_copy_logits']=self.copy_logits
  exec(compile('from __future__ import annotations\n'+(R/'method.copied.py.txt').read_text(),str(R/'method.copied.py.txt'),'exec'),ns)
  copied=ns['apply_sampling_params']
  @functools.wraps(old)
  def wrapper(owner,*args,**kwargs):return self.invoke(copied,owner,*args,**kwargs)
  self.bind(Sampler,'apply_sampling_params',wrapper)
  return dict(boundary='Sampler.apply_sampling_params BF16->FP32 copy',diagnostic_payloads=False,finite_guards_retained=True)
 def start(self):super().start();self.current=None;self.sampler_owner=None
 def copy_logits(self,x):
  self.guard();assert self.pending is not None and self.current is None
  import torch
  from tileir_vllm_sampler_fp32_copy_v1.api import validate_metadata
  validate_metadata(dict(shape=tuple(x.shape),stride=tuple(x.stride()),offset=x.storage_offset(),dtype=str(x.dtype)))
  assert x.untyped_storage().nbytes()==15892480 and not x.requires_grad
  assert bool(torch.isfinite(x).all()),'Finite workload scope only'
  y=self.call('sampler',self.pending['phase'],self.api,x)
  assert y.shape==x.shape and y.dtype==torch.float32 and y.device==x.device and y.stride()==(248320,1) and y.storage_offset()==0 and y.untyped_storage().nbytes()==31784960
  assert y.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()
  assert bool(torch.isfinite(y).all()),'Original output finite scope retained'
  self.current=y;return y
 def invoke(self,copied,owner,*args,**kwargs):
  self.guard();assert self.pending is None
  if self.sampler_owner is None:self.sampler_owner=owner
  assert owner is self.sampler_owner
  self.pending=dict(phase=phase_index(sum(self.counts.values())));self.current=None
  try:
   result=copied(owner,*args,**kwargs)
   assert self.current is not None and result is self.current,'Same post-processor buffer required'
   return result
  except BaseException as e:self.controller.latch(self.key,e);raise
  finally:self.pending=None;self.current=None
 def finish(self):
  r=super().finish();assert self.counts[('sampler','prefill')]==1 and self.counts[('sampler','decode')]==15
  r.update(native_copy_calls=0,finite_guards_retained=True,post_processor_mutation_allowed=True);return r
