import unittest,json,ast,sys,types
from pathlib import Path
from unittest.mock import patch
from .bindings import HiddenGather,SamplerFP32,SamplerFour,HiddenOriginal,CopyOriginal,Top5Qualified,SITES
from .gates import dispatch
R=Path(__file__).parent;P=R.parent
class T(unittest.TestCase):
 def part(self,cls):
  p=cls.__new__(cls);p.key='test';p.dead=False
  def guard():assert not p.dead,'latched'
  def latch(*a):p.dead=True
  p.guard=guard;p.controller=types.SimpleNamespace(latch=latch);return p
 def test_hidden_captured_bound_override(self):
  p=self.part(HiddenGather);captured=[]
  def install(part,model):captured.append(part.gather);return {}
  with patch.object(HiddenOriginal,'install',install):p.install(None)
  self.assertIs(captured[0],p.gather)
  self.assertIn('guarded_body',p.gather.__code__.co_names)
 def test_copy_owned_method_and_api_slots(self):
  from tileir_vllm_sampler_fp32_copy_v1.api import sampler_fp32_copy
  p=self.part(SamplerFP32);captured=[]
  def install(part,model):captured.append(part.copy_logits);part.api=sampler_fp32_copy;return {}
  with patch.object(CopyOriginal,'install',install):p.install(None)
  self.assertIs(captured[0],p.copy_logits);self.assertIsNot(p.api,sampler_fp32_copy)
  self.assertEqual(p.api.__globals__['_finite_all'],p.finite_all)
  self.assertNotIn('_finite_all',sampler_fp32_copy.__globals__)
 def test_owned_helper_fresh_global_bindings(self):
  from tileir_vllm_selected_ranks_v1 import selected_token_ranks
  from tileir_vllm_selected_logprobs_v2 import compute_token_logprobs
  p=self.part(SamplerFour)
  orig={'_tileir_selected_ranks':selected_token_ranks,'_tileir_selected_logprobs':compute_token_logprobs}
  def install(part,model):
   ns=dict(orig);exec('def helper(x,i):return _tileir_selected_logprobs(x,i),_tileir_selected_ranks(x,i)',ns);part.candidate=ns['helper'];return {}
  with patch.object(Top5Qualified,'install',install):p.install(None)
  for key,fn in orig.items():
   self.assertIsNot(p.candidate.__globals__[key],fn)
   self.assertEqual(p.candidate.__globals__[key].__globals__['_finite_all'],p.finite_all)
   self.assertNotIn('_finite_all',fn.__globals__)
 def test_six_census96_and_error_identity(self):
  total=0
  for cls in (HiddenGather,SamplerFP32,SamplerFour):
   p=self.part(cls);p.setup_finite();p.finite_impl_six=lambda x:None
   for site in p.site_names:
    for i in range(16):self.assertTrue(p.finite_all(site,None));total+=1
   self.assertTrue(all(v==[16,16]for v in p.finite_counts.values()))
  self.assertEqual(total,96)
  def bad(x):raise AssertionError('NaN/Inf outside finite-only contract')
  p.finite_impl_six=bad
  def body():assert p.finite_all(p.site_names[0],None),'original message'
  with self.assertRaisesRegex(AssertionError,'original message'):p.guarded_body(body)
  with self.assertRaisesRegex(AssertionError,'latched'):p.guarded_body(body)
 def test_real_result_extended_gate(self):
  d=json.loads((P/'lean_finite_guard_runtime_v1/finiteModel01/RESULT.json').read_text())['lean_request']['selected_dispatch']
  mapping={'hidden_gather':HiddenGather.site_names,'sampler_fp32':SamplerFP32.site_names,'sampler_four':SamplerFour.site_names}
  for key,sites in mapping.items():d['components'][key]['finite_sites']={s:dict(attempts=16,passes=16,original_selected_predicate_calls=0,host_status_item_retained=True)for s in sites}
  self.assertTrue(dispatch(d)['passed'])
  d['components']['hidden_gather']['finite_sites']['hidden_post']['passes']=15
  with self.assertRaises(AssertionError):dispatch(d)
 def test_source_only_ast(self):
  for f in R.glob('*.py'):ast.parse(f.read_text())
  self.assertFalse((R/'cache_v1').exists());self.assertFalse((R/'CACHE-PRESEED-01.json').exists())
  self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda.tile',sys.modules);self.assertNotIn('vllm',sys.modules)
if __name__=='__main__':unittest.main()
