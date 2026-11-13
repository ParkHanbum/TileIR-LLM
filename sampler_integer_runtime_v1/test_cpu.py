import unittest,types,sys,ast
from pathlib import Path
from unittest.mock import patch
from .integer import Integer
from . import adapter
R=Path(__file__).parent
class T(unittest.TestCase):
 def test_actual_owned_call16_live_helpers_nested_contexts(self):
  ns={'torch':types.SimpleNamespace(Tensor=object),'InputBatch':object,'SamplerOutput':lambda **kw:kw,'PROCESSED_LOGPROBS_MODES':set()}
  exec((R/'call.original.py.txt').read_text(),ns);S=type('Sampler',(),{'__call__':ns['__call__']})
  module=types.SimpleNamespace(Sampler=S,get_num_nans=lambda *a:None,compute_topk_scores=lambda *a,**k:None,get_num_sampled_and_rejected=lambda *a:(None,None))
  ns.update({key:getattr(module,key)for key in ('get_num_nans','compute_topk_scores','get_num_sampled_and_rejected')})
  p=Integer.__new__(Integer);p.model=types.SimpleNamespace();p.owner=None;p.wrapped=None;p.active=False;p.failed=None;p.rows=[];p.shadow_pass=False;p.mode=None
  def gather(site,*a):p.rows.append({'site':site});return site
  def ones(*a):p.rows.append({'site':'ones'});return 'ones'
  p.gather=gather;p.ones=ones
  fake={'vllm.v1.worker.gpu.sample':types.SimpleNamespace(sampler=module)}
  with patch.dict(sys.modules,fake),patch('inspect.getsource',return_value=(R/'call.original.py.txt').read_text()):p.install()
  # Later wrapper installation captures current owned body, with live helpers.
  pending=[];helper_calls=[];originals=[]
  for name in ('counts','greedy','four'):
   old=S.__call__;originals.append(old)
   def wrapper(owner,*a,_old=old,_name=name,**k):
    pending.append(_name)
    try:return _old(owner,*a,**k)
    finally:pending.pop()
   S.__call__=wrapper
  def helper(name,value):
   def fn(*a,**k):self.assertEqual(pending,['four','greedy','counts']);helper_calls.append(name);return value
   return fn
  module.compute_topk_scores=helper('topk','scores');module.get_num_sampled_and_rejected=helper('counts',('numsampled','numrejected'))
  sampled=types.SimpleNamespace(view=lambda *a:'sampled32x1')
  owner=S();owner.compute_nans=False;owner.get_logprobs_dims=lambda x:(5,0);owner.sample=helper('sample',(sampled,'processed'));owner.trace_replay_state=None;owner.logprobs_mode='raw_logprobs';owner.logprob_token_ids_state=None;owner.return_sampling_mask=False;owner.req_states=types.SimpleNamespace(prefill_len=types.SimpleNamespace(gpu='prefill'))
  batch=types.SimpleNamespace(expanded_idx_mapping='expanded',idx_mapping='idx',idx_mapping_np=types.SimpleNamespace(shape=(32,)),cu_num_logits_np=types.SimpleNamespace(tolist=lambda:[]),expanded_local_pos='local',positions='pos',input_ids='ids',logits_indices='indices',seq_lens_cpu_upper_bound=types.SimpleNamespace(numpy=lambda:[]),seq_lens='seq',num_reqs=32,cu_num_logits='cu')
  logits=types.SimpleNamespace(shape=(32,248320));p.start('shadow')
  for i in range(16):self.assertEqual(owner(logits,batch)['sampled_token_ids'],'sampled32x1')
  self.assertEqual([x['site']for x in p.rows],['positions','input_ids','ones']*16);self.assertEqual(helper_calls,['sample','topk','counts']*16)
  with self.assertRaises(AssertionError):owner(logits,batch)
  self.assertEqual(len(helper_calls),48)
  for old in reversed(originals):S.__call__=old
  for key,value in p.initial_helpers.items():setattr(module,key,value)
  self.assertTrue(p.close()['restored']);self.assertIs(S.__call__,ns['__call__'])
 def test_independent_cleanup(self):
  m=types.SimpleNamespace();events=[]
  with patch.object(adapter.lean,'close',side_effect=RuntimeError('fail')),patch.object(adapter.integer,'close',side_effect=lambda model:events.append('integer')or {'restored':True}):
   with self.assertRaises(RuntimeError):adapter.close(m)
  self.assertEqual(events,['integer'])
  with self.assertRaises(AssertionError):adapter.release(m)
 def test_other_component_failure_prevents_entry(self):
  from lean26_gdn_finite_runtime_v1.adapter import ATTR
  model=types.SimpleNamespace();setattr(model,ATTR,types.SimpleNamespace(failed='other failed'))
  p=Integer.__new__(Integer);p.model=model;p.active=True;p.failed=None;p.rows=[]
  with self.assertRaisesRegex(AssertionError,'Another component'):p.guard()
 def test_ast_sources(self):
  for f in R.glob('*.py'):ast.parse(f.read_text())
  for n in ('torch','vllm','cuda.tile'):self.assertNotIn(n,sys.modules)
  self.assertFalse((R/'cache_v1').exists())
if __name__=='__main__':unittest.main()
