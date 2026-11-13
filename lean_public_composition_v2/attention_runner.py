"""Lean public attention/runner bindings. CPU draft; requires independent GPU proof.
Only diagnostic tensor snapshots/oracles/hashes are removed. Numerical APIs and
supported-value guards remain, including their native kernels/host reads.
"""
import inspect,json,hashlib,textwrap
from pathlib import Path
from lean_public_composition_v1.support import Base,phase_index
P=Path(__file__).resolve().parent.parent

def disjoint(*ts):
 spans=[(t.untyped_storage().data_ptr(),t.untyped_storage().nbytes())for t in ts]
 for i,(a,n)in enumerate(spans):
  for b,m in spans[i+1:]:assert a+n<=b or b+m<=a,'Overlapping owners'

def owned_copy(original,folder,stem,bindings,function_name=None):
 r=P/folder;spec=json.loads((r/'SOURCE.json').read_text())
 assert hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()==spec['sha256']
 assert textwrap.dedent(inspect.getsource(original)).strip()==(r/(stem+'.original.py.txt')).read_text().strip()
 ns=dict(original.__globals__);ns.update(bindings)
 exec(compile('from __future__ import annotations\n'+(r/(stem+'.copied.py.txt')).read_text(),str(r/(stem+'.copied.py.txt')),'exec'),ns)
 return ns[function_name or stem]

class AttentionCore(Base):
 def install(self,model):
  from attention_scoped_runtime_v5.adapter import MODULES,classify,validate_scalars
  from tileir_paged_attention_v10.api import attention
  self.classify=classify;self.scalars=validate_scalars;self.candidate=attention;self.seen=set()
  found=[(n,m)for n,m in model.named_modules()if n.endswith('.self_attn.attn')];assert {n for n,m in found}==MODULES
  self.owners={n for n,m in found};mods={}
  for name,owner in found:
   impl=owner.impl;orig=impl.forward;assert type(impl).__name__=='FlashAttentionImpl';mod=__import__(orig.__module__,fromlist=['x']);mods[id(mod)]=mod;sig=inspect.signature(orig)
   def forward(*args,_orig=orig,_name=name,_sig=sig,**kw):
    assert self.pending is None
    b=_sig.bind(*args,**kw);b.apply_defaults();md=b.arguments['attn_metadata'];assert md is not None
    starts=md.query_start_loc.detach().cpu().tolist();lens=md.seq_lens.detach().cpu().tolist();L,supported=self.classify(starts,lens)
    assert supported and (_name,L)not in self.seen
    ctx=dict(name=_name,length=L,starts=starts,lens=lens,calls=0,out=b.arguments['output']);self.pending=ctx
    try:
     result=_orig(*args,**kw);assert ctx['calls']==1 and result is ctx['out'];self.seen.add((_name,L));return result
    finally:self.pending=None
   self.bind(impl,'forward',forward)
  for mod in mods.values():
   sig=inspect.signature(mod.flash_attn_varlen_func)
   def helper(*args,_sig=sig,**kw):
    assert self.pending is not None;ctx=self.pending;ctx['calls']+=1;assert ctx['calls']==1
    b=_sig.bind(*args,**kw);b.apply_defaults();return self.apply(b.arguments,ctx)
   self.bind(mod,'flash_attn_varlen_func',helper)
  return dict(owners=sorted(self.owners),scope='Actual shared KV/caller output; no native attention target')
 def apply(self,v,ctx):
  import torch
  self.scalars(v);q,k,val,out=(v[n]for n in('q','k','v','out'));cuq,lens,table=(v[n]for n in('cu_seqlens_q','seqused_k','block_table'))
  assert out is not None and out.data_ptr()==ctx['out'].data_ptr()
  assert cuq.cpu().tolist()==ctx['starts'] and lens.cpu().tolist()==ctx['lens']
  assert v['max_seqlen_q']==ctx['starts'][1]and v['max_seqlen_k']==ctx['length']
  inputs={n:t for n,t in v.items()if isinstance(t,torch.Tensor)and n!='out'}
  for t in inputs.values():assert t.device==q.device and not t.requires_grad;disjoint(t,out)
  for n in('q_descale','k_descale','v_descale'):
   t=v[n];assert t.dtype==torch.float32 and t.shape==(32,4)and t.stride()==(0,0)and bool((t==1).all())
  # Unchanged public API validates layouts, shared KV offset, block bounds and lengths.
  result=self.call(ctx['name'],'prefill'if ctx['length']==53 else'decode',self.candidate,q,k,val,cuq,lens,table,out,scale=.0625,causal=True)
  assert result is out and bool(torch.isfinite(out).all());return result
 def start(self):super().start();self.seen.clear()
 def finish(self):
  r=super().finish();assert self.seen=={(n,L)for n in self.owners for L in range(53,69)};return r

class RunnerPadding(Base):
 def install(self,model):
  import vllm.v1.worker.gpu.model_runner as mr
  self.positions=next(p for p in self.controller.parts if p.key=='pos_seq')
  matches=[b for b in self.positions.bindings if b[0]is mr.GPUModelRunner and b[1]=='prepare_inputs'];assert len(matches)==1
  original=matches[0][2];assert mr.GPUModelRunner.prepare_inputs is matches[0][5]
  self.original=original;self.kept=[];self.current=[];self.fill_current=[]
  copied=owned_copy(original,'runner_padding_runtime_integration_v1','prepare_inputs',{'_construct':self.construct,'_fill_padding':self.fill_padding})
  self.bind(mr.GPUModelRunner,'prepare_inputs',lambda runner,*a,**k:self.invoke(copied,runner,*a,**k))
  return dict(boundary='prepare_inputs',logical_operations=64,position_bridge='exact lean Positions binding')
 def construct(self,kind,n,device):
  import torch
  from tileir_vllm_runner_constructors_v1 import api
  assert self.pending is not None and len(self.current)<2 and kind==('cu_num_logits','expanded_local_pos')[len(self.current)]and type(n)is int and n==32
  tile=getattr(api,kind)(n,device);length=33 if kind=='cu_num_logits'else 32
  assert tile.dtype==torch.int32 and tile.shape==(length,)and tile.stride()==(1,)and tile.storage_offset()==0 and tile.untyped_storage().nbytes()==length*4 and tile.device==torch.device(device)
  for prior in self.kept:disjoint(tile,prior)
  self.kept.append(tile);self.current.append(tile);return tile
 def fill_padding(self,view,value):
  from tileir_vllm_runner_bool_fill_v1.api import fill_padding_,validate_contract
  assert self.pending is not None and len(self.fill_current)<2 and type(value)is bool and value is bool(len(self.fill_current))
  n=validate_contract(tuple(view.shape),tuple(view.stride()),str(view.dtype),view.storage_offset(),view.untyped_storage().nbytes(),value)
  assert view.untyped_storage().nbytes()<=65536 and ((not value and n in(32,1696))or(value and n==0))
  ret=fill_padding_(view,value);assert ret is view;self.fill_current.append(value);return ret
 def invoke(self,copied,runner,*a,**k):
  from pos_seq_runtime_capture_v2.observer import state_metadata
  assert self.pending is None and self.positions.pending is None
  bound=inspect.signature(self.original).bind(runner,*a,**k);bound.apply_defaults();s=bound.arguments['scheduler_output'];b=bound.arguments['batch_req_state']
  assert not s.scheduled_spec_decode_tokens and len(b.req_ids)==32 and runner.model_state.num_new_sampled_tokens_per_step==1 and runner.pcp_manager is None
  ctx=dict(owner=runner,batch=state_metadata(b),calls=0);self.positions.pending=ctx;self.pending=ctx;self.current=[];self.fill_current=[]
  try:
   out=copied(runner,*a,**k)
   assert ctx['calls']==1 and len(self.current)==2 and self.fill_current==[False,True]
   assert out.cu_num_logits is self.current[0]and out.expanded_local_pos is self.current[1]
   assert out.req_ids is b.req_ids and out.idx_mapping is out.expanded_idx_mapping and out.num_reqs==32 and out.num_draft_tokens==0
   self.counts[('runner',ctx['batch']['phase'])]+=1;return out
  finally:self.positions.pending=None;self.pending=None
 def start(self):super().start();self.kept=[];self.current=[];self.fill_current=[]
 def finish(self):
  r=super().finish();assert self.counts=={('runner','prefill'):1,('runner','decode'):15};r['logical_suboperation_calls']=64;return r

class RunnerCensus(Base):
 def finish(self):
  r=super().finish();assert self.counts=={('runner','prefill'):1,('runner','decode'):15};return r

class HiddenGather(RunnerCensus):
 def install(self,model):
  import vllm.v1.worker.gpu.model_runner as mr
  self.model=model;self.current=None;original=mr.GPUModelRunner.sample
  copied=owned_copy(original,'runner_hidden_gather_runtime_integration_v2','sample',{'_gather':self.gather})
  self.bind(mr.GPUModelRunner,'sample',lambda runner,*a,**k:self.invoke(copied,runner,*a,**k));return dict(boundary='sample/hidden_rows')
 def gather(self,x,idx):
  import torch
  from tileir_vllm_hidden_gather_v1 import gather_hidden_rows
  from tileir_vllm_hidden_gather_v1.api import validate_metadata
  assert self.pending is not None and self.current is None;n=1696 if self.pending=='prefill'else 32
  assert x.shape==(n,2560)and x.dtype==torch.bfloat16 and x.stride()==(2560,1)and x.storage_offset()==0 and x.untyped_storage().nbytes()==n*5120
  assert idx.shape==(32,)and idx.dtype==torch.int64 and idx.stride()==(1,)and idx.storage_offset()==0 and idx.untyped_storage().nbytes()==256
  meta=lambda t:dict(shape=tuple(t.shape),stride=tuple(t.stride()),offset=t.storage_offset(),dtype=str(t.dtype))
  validate_metadata(meta(idx),meta(x),idx.detach().cpu().tolist());disjoint(x,idx)
  y=self.call('runner',self.pending,gather_hidden_rows,x,idx)
  assert y.shape==(32,2560)and y.dtype==x.dtype and y.device==x.device and y.stride()==(2560,1)and y.storage_offset()==0 and y.untyped_storage().nbytes()==163840
  disjoint(y,x,idx);assert bool(torch.isfinite(y).all());self.current=y;return y
 def invoke(self,copied,runner,*a,**k):
  assert self.pending is None and runner.model is self.model and runner.batch_sharder is None
  self.pending=phase_index(sum(self.counts.values()));self.current=None
  try:ret=copied(runner,*a,**k);assert self.current is not None;return ret
  finally:self.pending=None

class AcceptedScatter(RunnerCensus):
 def install(self,model):
  import vllm.v1.worker.gpu.model_states.mamba_hybrid as m
  original=m.MambaHybridModelState.postprocess_state
  copied=owned_copy(original,'mamba_accepted_runtime_integration_v2','postprocess',{'_selected_scatter':self.scatter},'postprocess_state')
  self.bind(m.MambaHybridModelState,'postprocess_state',lambda state,idx_mapping,num_sampled,num_computed_tokens=None:self.invoke(copied,state,idx_mapping,num_sampled,num_computed_tokens))
  return dict(boundary='postprocess_state/scatter',alignment='off')
 def scatter(self,idx,sampled,dst):
  import torch
  from tileir_vllm_mamba_accepted_scatter_v1.api import scatter_num_accepted_,validate_indices
  assert self.pending is not None and self.pending['calls']==0;self.pending['calls']+=1
  assert idx.dtype==torch.int64 and sampled.dtype==dst.dtype==torch.int32 and idx.shape==sampled.shape==dst.shape==(32,)
  assert all(t.stride()==(1,)and t.storage_offset()==0 and t.numel()*t.element_size()<=65536 and t.untyped_storage().nbytes()<=65536 for t in(idx,sampled,dst))
  assert idx.device==sampled.device==dst.device;disjoint(idx,sampled,dst);validate_indices(idx.cpu().tolist(),32)
  ret=self.call('runner',self.pending['phase'],scatter_num_accepted_,idx,sampled,dst);assert ret is None;return None
 def invoke(self,copied,state,idx,sampled,computed=None):
  import torch
  assert self.pending is None and isinstance(idx,torch.Tensor)and isinstance(sampled,torch.Tensor)and idx.shape==sampled.shape==(32,)
  assert state.recoverssm is None and not state._align_mode
  self.pending=dict(phase=phase_index(sum(self.counts.values())),calls=0);dst=state.num_accepted_tokens_gpu
  try:ret=copied(state,idx,sampled,computed);assert ret is None and state.num_accepted_tokens_gpu is dst and self.pending['calls']==1;return ret
  finally:self.pending=None
