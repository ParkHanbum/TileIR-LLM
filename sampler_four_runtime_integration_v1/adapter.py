"""Public helper copy replaces top5 v2/ranks/ordered selected logprobs."""
from pathlib import Path
import inspect,json,hashlib
from block_gather_runtime_capture_v1.observer import phase_metadata
from tp1_embedding_runtime_capture_v2.observer import meta
from norm_runtime_contract_capture_v1.observer import identity,aliases
from block_gather_runtime_integration_v1.adapter import bytecopy
ATTR='_sampler_four_runtime_v1';ROOT=Path(__file__).resolve().parent
def valid_counts(counts):return len(counts)==5 and all(len(col)==32 and all(type(n)is int and n>=1 for n in col) for col in counts)
def census(rows):return len(rows)==16 and [x['phase'] for x in rows]==['prefill']+['decode']*15
def build_copy(module,rank_fn,top5_fn,score_fn,cat_fn):
 spec=json.loads((ROOT/'SOURCE.json').read_text());assert hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()==spec['sha256']
 source=(ROOT/'copied_scores.py.txt').read_bytes();assert hashlib.sha256(source).hexdigest()==spec['copy_sha256']
 env=dict(module.__dict__,_tileir_selected_ranks=rank_fn,_tileir_top5=top5_fn,_tileir_selected_logprobs=score_fn,_tileir_cat=cat_fn);exec(compile(source,str(ROOT/'copied_scores.py.txt'),'exec'),env);return env['compute_topk_scores']
class Adapter:
 def __init__(self,output):
  self.output=Path(output);self.output.mkdir(exist_ok=False);self.bindings=[];self.pending=None;self.rows=[];self.mode='shadow';self.shadow_pass=False;self.native_gate_calls=0;self.tileir_gate_calls=0;self.first_mismatch=None;self.failed=False;self.cat_rows=[];self.cat_native_calls=0;self.cat_tile_calls=0
 def install_classes(self,S,module):
  original_call=S.__call__;fn=module.compute_topk_scores
  def call(owner,logits,batch,*args,**kwargs):
   assert not self.failed,'Latched first mismatch; no queued native/candidate work'
   assert self.pending is None
   ctx=dict(batch=phase_metadata(batch),calls=0);self.pending=ctx
   try:
    assert owner.logprobs_mode=='raw_logprobs' and owner.trace_replay_state is None
    ret=original_call(owner,logits,batch,*args,**kwargs);assert ctx['calls']==1;return ret
   except BaseException:self.failed=True;raise
   finally:self.pending=None
  def helper(*args,**kwargs):
   assert not self.failed,'Latched first mismatch'
   ctx=self.pending;assert ctx is not None;ctx['calls']+=1;assert ctx['calls']==1
   bound=inspect.signature(fn).bind(*args,**kwargs);bound.apply_defaults()
   try:return self.apply(bound.arguments,lambda:fn(*args,**kwargs),ctx['batch'])
   except BaseException:self.failed=True;raise
  for obj,name,old,new in [(S,'__call__',original_call,call),(module,'compute_topk_scores',fn,helper)]:
   self.bindings.append((obj,name,old,new));setattr(obj,name,new)
 def install(self,model):
  from vllm.v1.worker.gpu.sample import sampler,logprob
  from tileir_vllm_selected_ranks_v1 import selected_token_ranks
  from tileir_vllm_top5_v2 import top5_indices
  from tileir_vllm_selected_logprobs_v2 import compute_token_logprobs
  assert sampler.compute_topk_scores is logprob.compute_topk_scores
  source=identity(sampler.compute_topk_scores);spec=json.loads((ROOT/'SOURCE.json').read_text());assert source['source_sha256']==spec['sha256']
  self.copy=build_copy(logprob,selected_token_ranks,top5_indices,compute_token_logprobs,self.cat);self.install_classes(sampler.Sampler,sampler)
  return dict(source=source,copy_sha256=spec['copy_sha256'],scope='Top5 v2 and ranks functional copy; ordered selected-logprobs; TileIR cat; native output object construction retained')
 def cat(self,chosen,top5):
  assert not self.failed,'Latched first cat mismatch'
  import torch
  from tileir_vllm_sampler_cat_v1 import concat_sampled_top5
  try:
   assert self.pending is not None and len(self.cat_rows)==len(self.rows) and len(self.cat_rows)<16
   inputs={'chosen':chosen,'top5':top5};before={k:bytecopy(t) for k,t in inputs.items()};descs={k:meta(t) for k,t in inputs.items()}
   native=None
   if self.mode=='shadow':native=torch.cat((chosen,top5),dim=1);self.cat_native_calls+=1
   candidate=concat_sampled_top5(chosen,top5);self.cat_tile_calls+=1
   expected=torch.cat((before['chosen'].view(torch.int64).reshape(32,1),before['top5'].view(torch.int64).reshape(32,5)),dim=1)
   actual=candidate.detach().cpu();raw=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
   native_raw=native is None or torch.equal(bytecopy(native),bytecopy(candidate))
   layout=candidate.dtype==torch.int64 and tuple(candidate.shape)==(32,6) and candidate.stride()==(6,1) and candidate.storage_offset()==0 and candidate.untyped_storage().nbytes()==1536
   preserved=all(torch.equal(before[k],bytecopy(t)) for k,t in inputs.items())
   fresh=all(not z['storage_overlap'] for z in aliases({**inputs,'output':candidate}).values())
   row=dict(mode=self.mode,inputs=descs,raw_equal=raw,native_raw_equal=native_raw,integer_exact=raw,layout=layout,input_full_storage_preserved=preserved,fresh_disjoint=fresh,passed=raw and native_raw and layout and preserved and fresh);self.cat_rows.append(row)
   if not row['passed']:
    self.failed=True;bank=self.output/'first_cat_mismatch';bank.mkdir(exist_ok=False)
    torch.save(dict(inputs=before,native=None if native is None else native.cpu(),candidate=actual,expected=expected),bank/'values.pt');self.first_mismatch=dict(cat=row,path=str(bank/'values.pt'));self.save();raise AssertionError('Cat raw/contract HOLD')
   return candidate
  except BaseException:
   self.failed=True;raise
 def apply(self,v,native,batch):
  assert not self.failed,'Latched first mismatch; no queued native/candidate work'
  import torch
  from tileir_vllm_selected_ranks_v1 import selected_token_ranks
  from tileir_vllm_selected_ranks_v1.api import validate_descriptors
  from gdn_post_conv_qualification_v2.run import compare
  assert self.mode in ('shadow','replace') and (self.mode!='replace' or self.shadow_pass)
  assert len(self.rows)<16 and batch['num_reqs']==32 and batch['phase']==('prefill' if len(self.rows)==0 else 'decode')
  assert v['num_logprobs']==5 and v['max_per_req_token_ids']==0 and v['logits_mode'] is False and v['cu_num_logits'] is None
  x,ids=v['logits'],v['sampled_token_ids'];validate_descriptors(meta(x),meta(ids))
  tensors={k:t for k,t in v.items() if isinstance(t,torch.Tensor)};before={k:bytecopy(t) for k,t in tensors.items()}
  if self.mode=='shadow':
   ret=native();self.native_gate_calls+=1;candidate_ret=self.copy(**v);candidate=candidate_ret.selected_token_ranks;candidate_scores=candidate_ret.logprobs
  else:ret=self.copy(**v);candidate_ret=ret;candidate=ret.selected_token_ranks;candidate_scores=ret.logprobs
  self.tileir_gate_calls+=1;torch.cuda.synchronize();rank=ret.selected_token_ranks
  raw=True if self.mode=='replace' else torch.equal(rank.view(torch.uint8),candidate.view(torch.uint8))
  layout=all(t.dtype==torch.int64 and tuple(t.shape)==(32,) and t.stride()==(1,) and t.storage_offset()==0 for t in (rank,candidate))
  valid=all(bool(((t>=1)&(t<=248320)).all()) for t in (rank,candidate))
  preserved=all(torch.equal(before[k],bytecopy(t)) for k,t in tensors.items())
  fresh=all(not z['storage_overlap'] for y in (rank,candidate) for z in aliases({**tensors,'rank':y}).values())
  prefix=ret.logprob_token_ids.shape==(32,6) and ret.logprobs.shape==(32,6) and ret.logprobs.dtype==torch.float32 and bool(torch.isfinite(ret.logprobs).all()) and torch.equal(ret.logprob_token_ids[:,0],ids)
  score_comparison=compare(ret.logprobs,candidate_scores,torch)
  score_layout=candidate_scores.dtype==torch.float32 and candidate_scores.shape==(32,6) and candidate_scores.stride()==(6,1) and candidate_scores.storage_offset()==0
  score_fresh=all(not z['storage_overlap'] for z in aliases({**tensors,'scores':candidate_scores}).values())
  score_pass=score_layout and score_fresh and score_comparison['raw_equal'] and score_comparison['strict_failures']==0 and score_comparison['candidate_nonfinite']==score_comparison['reference_nonfinite']==0
  candidate_ids=candidate_ret.logprob_token_ids
  ids_raw=torch.equal(ret.logprob_token_ids.view(torch.uint8),candidate_ids.view(torch.uint8))
  ids_layout=candidate_ids.dtype==torch.int64 and candidate_ids.shape==(32,6) and candidate_ids.stride()==(6,1) and candidate_ids.storage_offset()==0
  ids_fresh=all(not z['storage_overlap'] for z in aliases({**tensors,'selected_ids':candidate_ids}).values())
  selected_values=x.gather(1,candidate_ids[:,1:]);tie_counts=[(x==selected_values[:,j,None]).sum(dim=1).cpu().tolist() for j in range(5)]
  tie_count_contract=valid_counts(tie_counts)
  passed=raw and layout and valid and preserved and fresh and bool(prefix) and score_pass and ids_raw and ids_layout and ids_fresh and tie_count_contract
  row=dict(mode=self.mode,phase=batch['phase'],batch=batch,inputs={k:meta(t) for k,t in tensors.items()},raw_integer_equal=raw if self.mode=='shadow' else None,raw_reference='same-call native rank' if self.mode=='shadow' else None,layout_equal=layout,input_full_storage_preserved=preserved,fresh_disjoint=fresh,output_valid_range=valid,native_topk_retained=False,top5_ids_raw_equal=ids_raw if self.mode=='shadow' else None,top5_ids_layout=ids_layout,top5_ids_fresh=ids_fresh,top5_tie_counts=tie_counts,tie_count_contract=tie_count_contract,selected_logprob_comparison=score_comparison if self.mode=='shadow' else None,selected_logprob_layout=score_layout,selected_logprob_fresh=score_fresh,passed=passed)
  self.rows.append(row)
  if not passed:
   self.failed=True
   bank=self.output/'first_mismatch';bank.mkdir();torch.save(dict(inputs=before,native_or_actual=rank.cpu(),candidate=candidate.cpu(),native_or_actual_scores=ret.logprobs.cpu(),candidate_scores=candidate_scores.cpu(),native_or_actual_ids=ret.logprob_token_ids.cpu(),candidate_ids=candidate_ids.cpu()),bank/'values.pt');self.first_mismatch=dict(row=row,path=str(bank/'values.pt'),sha256=hashlib.sha256((bank/'values.pt').read_bytes()).hexdigest())
  self.save();assert passed,'Selected rank contract/raw HOLD';return ret
 def start(self,mode):
  assert not self.failed,'Latched first mismatch'
  assert mode in ('shadow','replace') and (mode!='replace' or self.shadow_pass)
  self.mode=mode;self.rows=[];self.cat_rows=[];self.cat_native_calls=self.cat_tile_calls=0;self.native_gate_calls=self.tileir_gate_calls=0;return dict(mode=mode)
 def summary(self):return dict(mode=self.mode,rows=self.rows,supported_calls=len(self.rows),native_gate_calls=self.native_gate_calls,tileir_gate_calls=self.tileir_gate_calls,shadow_pass=self.shadow_pass,first_mismatch=self.first_mismatch,supported_vendor_fallback_calls=0,suboperation_counts=dict(rank_native=self.native_gate_calls,rank_tileir=self.tileir_gate_calls,selected_logprob_native=self.native_gate_calls,selected_logprob_tileir=self.tileir_gate_calls,top5_native=self.native_gate_calls,top5_tileir=self.tileir_gate_calls),logical_suboperation_calls=4*len(self.rows),cat_rows=self.cat_rows,cat_native_validation_calls=self.cat_native_calls,cat_tileir_calls=self.cat_tile_calls,scope='supported_calls counts helper boundaries; four TileIR logical suboperations per helper; cat native_validation counts extra shadow audit calls, separate from native whole-helper cats')
 def save(self):(self.output/(self.mode+'-PROGRESS.json')).write_text(json.dumps(self.summary(),indent=2)+'\n')
 def finish(self):
  assert not self.failed and len(self.cat_rows)==16 and all(x['passed'] for x in self.cat_rows) and self.cat_tile_calls==16 and self.cat_native_calls==(16 if self.mode=='shadow' else 0)
  assert census(self.rows) and all(x['passed'] for x in self.rows) and self.tileir_gate_calls==16 and self.native_gate_calls==(16 if self.mode=='shadow' else 0)
  if self.mode=='shadow':self.shadow_pass=True
  self.save();return self.summary()
 def close(self):
  errors=[]
  for obj,name,old,new in reversed(self.bindings):
   try:assert getattr(obj,name) is new;setattr(obj,name,old)
   except BaseException as e:errors.append(repr(e))
  self.bindings=[];assert not errors,errors;return dict(restored=True)
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
