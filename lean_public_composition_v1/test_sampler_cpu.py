"""CPU-only wrapper/guard tests; no candidate numerical kernel executes."""
import ast,types,unittest
from pathlib import Path
from unittest.mock import patch
from . import sampler as a
class Controller:
 def __init__(self):self.active=True;self.failed=None
 def latch(self,key,error):
  if self.failed is None:self.failed=(key,repr(error))
class NestedTests(unittest.TestCase):
 def exercise(self,fail=False):
  events=[];ret=object();sampled=object();pair=(object(),object())
  def greedy(logits):raise AssertionError('native greedy forbidden')
  def counts(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len):raise AssertionError('native counts forbidden')
  def scores(logits):raise AssertionError('native scores forbidden')
  mod=types.SimpleNamespace(gumbel_sample=greedy,get_num_sampled_and_rejected=counts,compute_topk_scores=scores)
  class S:
   logprobs_mode='raw_logprobs';trace_replay_state=None
   def __call__(self,x,b):
    events.append('sampler')
    self.assert_g=mod.gumbel_sample(x);self.assert_c=mod.get_num_sampled_and_rejected(1,2,3,4,5)
    return mod.compute_topk_scores(x)
  old=S.__call__;c=Controller();cs=a.Counts(c,'sample_counts',16);gr=a.Greedy(c,'greedy_sample',16);sp=a.SamplerThree(c,'sampler_three',16)
  def mock_apply(obj,label,value):
   def apply(vals,ctx):
    def candidate():
     events.append(label)
     if fail and label=='scores':raise ValueError('candidate fails')
     return value
    return obj.call('sampler',ctx['metadata']['phase'],candidate)
   return apply
  cs.apply=mock_apply(cs,'counts',pair);gr.apply=mock_apply(gr,'greedy',sampled);sp.apply=mock_apply(sp,'scores',ret)
  with patch.object(a,'phase_metadata',lambda b:dict(num_reqs=32,phase=b)):
   cs.install_classes(S,mod);cw=S.__call__;gr.install_classes(S,mod);gw=S.__call__;sp.install_classes(S,mod)
   try:
    if fail:
     with self.assertRaises(ValueError):S()(object(),'prefill')
     before=list(events)
     with self.assertRaises(AssertionError):S()(object(),'prefill')
     self.assertEqual(events,before);self.assertIsNotNone(c.failed)
    else:
     for i in range(16):self.assertIs(S()(object(),'prefill'if i==0 else'decode'),ret)
     self.assertEqual(cs.finish()['calls'],16);self.assertEqual(gr.finish()['calls'],16);self.assertEqual(sp.finish()['logical_suboperation_calls'],48)
    for obj in(cs,gr,sp):self.assertIsNone(obj.pending)
   finally:
    sp.close();self.assertIs(S.__call__,gw);gr.close();self.assertIs(S.__call__,cw);cs.close()
  self.assertIs(S.__call__,old);self.assertIs(mod.gumbel_sample,greedy);self.assertIs(mod.compute_topk_scores,scores);self.assertIs(mod.get_num_sampled_and_rejected,counts)
 def test_three_wrappers_no_native_and_reverse_restore(self):self.exercise()
 def test_exception_latch_pending_cleanup(self):self.exercise(True)
 def test_wrong_batch_fails_before_any_helper(self):
  c=Controller();o=a.Greedy(c,'greedy_sample',16);called=[]
  class S:
   logprobs_mode='raw_logprobs';trace_replay_state=None
   def __call__(self,*args):called.append(True)
  m=types.SimpleNamespace(gumbel_sample=lambda logits:None)
  with patch.object(a,'phase_metadata',lambda b:dict(num_reqs=31,phase='prefill')):
   o.install_classes(S,m)
   try:
    with self.assertRaises(AssertionError):S()(None,None)
    self.assertFalse(called);self.assertIsNotNone(c.failed)
   finally:o.close()
 def test_unassociated_helper_fails_without_fallback(self):
  c=Controller();o=a.SamplerThree(c,'sampler_three',16);called=[]
  class S:
   def __call__(self,*args):pass
  m=types.SimpleNamespace(compute_topk_scores=lambda logits:called.append(True));o.install_classes(S,m)
  try:
   with self.assertRaises(AssertionError):m.compute_topk_scores(None)
   self.assertFalse(called);self.assertIsNotNone(c.failed)
  finally:o.close()
class ApplyTests(unittest.TestCase):
 def counts_fixture(self):
  class T:
   shape=(32,)
   def data_ptr(self):return 321
   def is_pinned(self):return True
  class UvaBuffer:
   def __init__(self):self.cpu=T();self._uva=T()
  active=UvaBuffer();owner=types.SimpleNamespace(pool=types.SimpleNamespace(_uva_bufs=[active],_curr=0),gpu=T());batch=types.SimpleNamespace(seq_lens=T(),cu_num_logits=T(),idx_mapping=T());c=Controller();o=a.Counts(c,'sample_counts',16)
  v=dict(num_sampled=T(),seq_lens=batch.seq_lens,cu_num_logits=batch.cu_num_logits,idx_mapping=batch.idx_mapping,prefill_len=owner.gpu);ctx=dict(owner=types.SimpleNamespace(req_states=types.SimpleNamespace(prefill_len=owner)),batch=batch,metadata=dict(phase='prefill'))
  return c,o,v,ctx,active
 def test_counts_actual_arguments_alias_and_owner_lifetime(self):
  c,o,v,ctx,active=self.counts_fixture();seen=[];rejected=object()
  def candidate(*args):seen.append(args);return(args[0],rejected)
  o.candidate=candidate;out=o.apply(v,ctx)
  self.assertIs(out[0],v['num_sampled']);self.assertIs(out[1],rejected)
  for x,k in zip(seen[0],('num_sampled','seq_lens','cu_num_logits','idx_mapping','prefill_len')):self.assertIs(x,v[k])
  self.assertIs(c.sampler_owner_refs[0][0],active)
 def test_counts_wrong_uva_owner_rejected(self):
  c,o,v,ctx,active=self.counts_fixture();o.candidate=lambda *args:self.fail('must not call');v['prefill_len']=object()
  with self.assertRaises(AssertionError):o.apply(v,ctx)
 def test_three_actual_argument_identity_and_scope(self):
  c=Controller();o=a.SamplerThree(c,'sampler_three',16);x=object();ids=object();ret=object();seen=[]
  v=dict(logits=x,sampled_token_ids=ids,num_logprobs=5,max_per_req_token_ids=0,logits_mode=False,cu_num_logits=None)
  def metadata(t):return dict(shape=[32,248320]if t is x else[32],stride=[248320,1]if t is x else[1],storage_offset=0,dtype='torch.bfloat16'if t is x else'torch.int64',requires_grad=False)
  o.candidate=lambda **kw:(seen.append(kw),ret)[1]
  with patch.object(a,'meta',metadata):
   self.assertIs(o.apply(v,dict(metadata=dict(phase='prefill'))),ret);self.assertIs(seen[0]['logits'],x);self.assertIs(seen[0]['sampled_token_ids'],ids)
   v['num_logprobs']=6
   with self.assertRaises(AssertionError):o.apply(v,dict(metadata=dict(phase='decode')))
   self.assertEqual(len(seen),1)
 def test_greedy_temperature_scope_cpu_only(self):
  import torch
  from tileir_vllm_greedy_sample_v1.api import validate_descriptor
  c=Controller();o=a.Greedy(c,'greedy_sample',16);x=object();seen=[];ret=object()
  v=dict(logits=x,expanded_idx_mapping=torch.arange(32,dtype=torch.int64),temperature=torch.zeros(32),seed=torch.zeros(32,dtype=torch.int64),pos=torch.zeros(32,dtype=torch.int64),apply_temperature=False,is_drafting=False,use_fp64=False,logits_cache=None,logits_cache_col=None,logits_cache_source=None)
  o.candidate=lambda actual:(seen.append(actual),ret)[1]
  metadata=dict(shape=[32,248320],stride=[248320,1],storage_offset=0,dtype='torch.float32',requires_grad=False)
  with patch.object(a,'meta',lambda t:metadata),patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA forbidden')):
   self.assertIs(o.apply(v,dict(metadata=dict(phase='prefill'))),ret);self.assertIs(seen[0],x)
   v['temperature'][3]=1
   with self.assertRaises(AssertionError):o.apply(v,dict(metadata=dict(phase='decode')))
   self.assertEqual(len(seen),1)
class SourceTests(unittest.TestCase):
 def test_no_external_snapshots_sync_or_numeric_change(self):
  text=Path(a.__file__).read_text();ast.parse(text)
  for forbidden in ('synchronize(','.cpu(','.clone(','.write_text(', 'torch.save(', 'bytecopy(', 'torch.equal('):self.assertNotIn(forbidden,text)
  self.assertIn('build_copy(logprob,selected_token_ranks,top5_indices,compute_token_logprobs)',text)
if __name__=='__main__':unittest.main()
