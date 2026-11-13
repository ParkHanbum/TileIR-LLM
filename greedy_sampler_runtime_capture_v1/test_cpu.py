import unittest,tempfile,types,json,sys
from pathlib import Path
from unittest.mock import patch
from greedy_sampler_runtime_capture_v1 import observer as o
from greedy_sampler_runtime_capture_v1.preflight import preflight
from greedy_sampler_runtime_capture_v1.result_gate import audit
class Tests(unittest.TestCase):
 def exercise(self,fail=False):
  events=[];sampled=object();top=object();result=object()
  def gumbel(*a,**kw):events.append('gumbel');return sampled
  def topk(*a,**kw):events.append('topk');return top
  mod=types.SimpleNamespace(gumbel_sample=gumbel,compute_topk_scores=topk)
  class S:
   logprobs_mode='raw_logprobs';trace_replay_state=None
   def __call__(self,logits,batch):
    assert mod.gumbel_sample(logits) is sampled
    if fail:raise ValueError('native error')
    assert mod.compute_topk_scores(logits) is top
    return result
  old=S.__call__
  with tempfile.TemporaryDirectory() as d:
   ob=o.Observer(Path(d)/'bank')
   def invoke(kind,fn,args,kwargs):
    ob.pending['counts'][kind]+=1;ob.pending['helpers'].append({'kind':kind});return fn(*args,**kwargs)
   ob.invoke=invoke
   with patch.object(o,'meta',lambda x:{}),patch.object(o,'identity',lambda x:{'name':x.__name__}),patch.object(o,'phase_metadata',lambda b:{'phase':b,'num_reqs':32}):
    ob.install_classes(S,mod)
    if fail:
     with self.assertRaises(ValueError):S()(None,'prefill')
     self.assertIsNone(ob.pending)
    else:
     for i in range(16):self.assertIs(S()(None,'prefill' if i==0 else 'decode'),result)
     self.assertEqual(len(ob.rows),16);self.assertEqual(events,['gumbel','topk']*16)
    ob.close();self.assertIs(S.__call__,old);self.assertIs(mod.gumbel_sample,gumbel);self.assertIs(mod.compute_topk_scores,topk)
 def test_16_once_return_restore(self):self.exercise()
 def test_exception_restore(self):self.exercise(True)
 def test_preflight(self):
  preflight(Path(__file__).parent)
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(AssertionError):preflight(d)
 def test_gate_failclosed(self):self.assertFalse(audit({})['passed'])
 def test_public_boundary_source(self):
  source=Path(__file__).with_name('observer.py').read_text()
  self.assertIn('size<=64*1024*1024',source);self.assertLess(source.index('size<=64*1024*1024'),source.index('c=t.detach().cpu()'))
  self.assertIn("values['sampled_token_ids'] is ctx['sampled']",source)
  self.assertNotIn('setattr(torch',source)
if __name__=='__main__':unittest.main()
