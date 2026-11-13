import unittest,ast,json,types,sys
from pathlib import Path
from unittest.mock import patch
from .sampler import SamplerFour,Original
from .adapter import SPECS,ORDER
from .gates import dispatch
R=Path(__file__).parent;P=R.parent
class T(unittest.TestCase):
 def part(self):
  a=SamplerFour.__new__(SamplerFour);a.key='sampler_four';a.dead=False
  def guard():assert not a.dead,'latched'
  def latch(*x):a.dead=True
  a.guard=guard;a.controller=types.SimpleNamespace(latch=latch);a.guard_attempts=a.guard_passes=a.top5_attempts=a.top5_returns=0;return a
 def test_counts(self):
  a=self.part();a.finite_impl=lambda x:None
  def target(x):a.finite_guard(x);return x
  a.guarded_impl=target
  for i in range(16):self.assertEqual(a.top5(i),i)
  with patch.object(Original,'finish',return_value={}):r=a.finish()
  self.assertEqual(r['finite_guard']['passes'],16);self.assertTrue(r['finite_guard']['host_scalar_status_retained'])
 def test_fail_latch_before_target(self):
  a=self.part();writes=[]
  def reject(x):raise AssertionError('NaN/Inf outside finite-only contract')
  a.finite_impl=reject
  def target(x):a.finite_guard(x);writes.append(1)
  a.guarded_impl=target
  with self.assertRaises(AssertionError):a.top5(None)
  with self.assertRaises(AssertionError):a.top5(None)
  self.assertEqual(writes,[]);self.assertEqual(a.guard_attempts,1);self.assertEqual(a.top5_returns,0)
 def test_owned_install_slot(self):
  a=self.part();old=lambda x:x
  def install(part,model):
   ns={'_tileir_top5':old};exec('def f(x):return _tileir_top5(x)',ns);part.candidate=ns['f'];return {}
  fake={'tileir_vllm_top5_v2':types.SimpleNamespace(top5_indices=old),'tileir_validation_primitives_v2.api':types.SimpleNamespace(require_finite=lambda x:None)}
  with patch.object(Original,'install',install),patch.dict(sys.modules,fake):result=a.install(None)
  self.assertTrue(result['finite_guard_replaced']);self.assertEqual(a.candidate.__globals__['_tileir_top5'],a.top5)
 def test_installed_actual_owned_helper_flow16(self):
  a=self.part();launches=[];old=lambda x:(_ for _ in ()).throw(AssertionError('Old top5 must not run'))
  class Tensor:
   shape=(32,248320);dtype='torch.bfloat16';requires_grad=False;device=types.SimpleNamespace(type='cuda')
   def stride(self):return (248320,1)
   def storage_offset(self):return 0
   def unsqueeze(self,*args):return self
  torch=types.SimpleNamespace(Tensor=Tensor,int32='int32',int64='int64',empty=lambda *a,**kw:object(),cuda=types.SimpleNamespace(current_stream=lambda *a:object()))
  tile=types.SimpleNamespace(launch=lambda *a:launches.append(a));kernels=types.SimpleNamespace(blocks=object(),finish=object())
  def install(part,model):
   ns=dict(torch=torch,LogprobsTensors=types.SimpleNamespace,_tileir_top5=old,_tileir_cat=lambda *a:object(),_tileir_selected_logprobs=lambda *a:object(),_tileir_selected_ranks=lambda *a:object())
   exec(compile('from __future__ import annotations\n'+(P/'sampler_four_runtime_integration_v1/copied_scores.py.txt').read_text(),'actual-owned-helper','exec'),ns);part.candidate=ns['compute_topk_scores'];return {}
  fake={'torch':torch,'cuda':types.SimpleNamespace(tile=tile),'cuda.tile':tile,'tileir_vllm_top5_v2':types.SimpleNamespace(top5_indices=old),'tileir_vllm_top5_v2.kernels':kernels,'tileir_validation_primitives_v2.api':types.SimpleNamespace(require_finite=lambda x:None)}
  with patch.object(Original,'install',install),patch.dict(sys.modules,fake):
   a.install(None)
   for i in range(16):a.candidate(Tensor(),5,Tensor())
  self.assertEqual(a.guard_attempts,16);self.assertEqual(a.guard_passes,16);self.assertEqual(a.top5_returns,16);self.assertEqual(len(launches),32)
 def test_same_order_one_class(self):
  from lean_public_composition_v2.adapter import SPECS as prior
  self.assertEqual([k for k,c,n in SPECS],[k for k,c,n in prior]);self.assertEqual(sum(c is not pc for (_,c,_),(_,pc,_)in zip(SPECS,prior)),1)
 def test_real_dispatch_gate(self):
  data=json.loads((P/'lean_public_runtime_qualification_v2/leanCorrectness02/RESULT.json').read_text());d=data['lean_request']['selected_dispatch'];d['components']['sampler_four']['finite_guard']=dict(attempts=16,passes=16,top5_api_returns=16,original_torch_finite_calls=0,host_scalar_status_retained=True)
  self.assertTrue(dispatch(d)['passed']);d['components']['sampler_four']['finite_guard']['passes']=15
  with self.assertRaises(AssertionError):dispatch(d)
 def test_delta_and_ast(self):
  old=(P/'finite_guard_qualification_v1/candidate.py').read_text();expected=old.replace(' from tileir_validation_primitives_v2.api import require_finite\n require_finite(logits)',' _finite_guard(logits)');self.assertEqual((R/'guarded_top5.py.txt').read_text(),expected)
  for p in R.glob('*.py'):ast.parse(p.read_text())
  self.assertNotIn('cuda.tile',sys.modules)
if __name__=='__main__':unittest.main()
