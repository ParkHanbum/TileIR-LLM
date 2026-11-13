import unittest,types,sys,json,ast,copy
from pathlib import Path
from unittest.mock import patch
from .core import PrefillCore,Original,ORDER,FLOAT,INTEGER
from .adapter import SPECS
from .gates import dispatch,qualification
R=Path(__file__).parent;P=R.parent
class Tensor:
 shape=(1,);dtype='fp32'
 def stride(self):return (1,)
 def storage_offset(self):return 0
 def numel(self):return 1
class T(unittest.TestCase):
 def part(self):
  c=types.SimpleNamespace(active=True,failed=None)
  def latch(k,e):
   if c.failed is None:c.failed=(k,e)
  c.latch=latch;p=PrefillCore(c,'gdn_prefill_core',24)
  with patch.object(Original,'install',return_value={}):p.install(None)
  return p
 def test_local_copy_and_composition(self):
  p=self.part();self.assertIsInstance(p,Original);self.assertEqual(p.body.__globals__['_gdn_finite'],p.finite)
  self.assertNotIn('_gdn_finite',Original.apply.__globals__)
  d={k:c for k,c,n in SPECS};self.assertEqual(len(d),26);self.assertIs(d['gdn_prefill_core'],PrefillCore)
  from lean_finite_guard_runtime_v4.bindings import SamplerFour,SamplerFP32,HiddenGather
  self.assertIs(d['sampler_four'],SamplerFour);self.assertIs(d['sampler_fp32'],SamplerFP32);self.assertIs(d['hidden_gather'],HiddenGather)
  from lean26_prefill_composition_v1.prefill import PrefillInputs
  self.assertIs(SPECS[0][1],PrefillInputs)
 def test24_ordered_paths360_and72(self):
  p=self.part();p.owners=set(str(i)for i in range(24));calls=[]
  torch=types.SimpleNamespace(bfloat16='bf16',float32='fp32',int32='i32',int64='i64',isfinite=lambda t:types.SimpleNamespace(all=lambda:True))
  def body(owner,v):
   for site in ORDER:
    t=Tensor();t.dtype='i32'if site in INTEGER else'fp32';assert owner.finite(t)
   return 'unchanged endpoint'
  p.body=body
  fake={'torch':torch,'tileir_validation_primitives_v3.api':types.SimpleNamespace(all_finite=lambda x:calls.append(1)or True)}
  with patch.dict(sys.modules,fake):
   for i in range(24):self.assertEqual(p.call(str(i),'prefill',p.apply,{}),'unchanged endpoint')
  result=p.finish();self.assertEqual(len(calls),360);self.assertEqual(result['gdn_finite']['retained_integer_checks'],72)
  with self.assertRaises(AssertionError):p.call('extra','prefill',p.apply,{})
 def test_error_latch_and_no_later_body(self):
  p=self.part();called=[]
  def fail(*a):called.append(1);raise AssertionError('original failure')
  p.body=fail
  with self.assertRaisesRegex(AssertionError,'original failure'):p.apply({})
  with self.assertRaises(AssertionError):p.apply({})
  self.assertEqual(called,[1]);self.assertIsNone(p.finite_scope)
 def test_dispatch_combined_negative(self):
  d=json.loads((P/'lean26_prefill_qualification_v1/lean26Prefill01/RESULT.json').read_text())['lean_request']['selected_dispatch']
  finite=json.loads((P/'lean_finite_guard_runtime_v4/finiteSixModel02/RESULT.json').read_text())['lean_request']['selected_dispatch']
  for key in ('hidden_gather','sampler_fp32','sampler_four'):
   d['components'][key]=finite['components'][key]
  d['components']['gdn_prefill_core']['gdn_finite']=dict(float_site_counts=dict.fromkeys(FLOAT,24),retained_integer_site_counts=dict.fromkeys(INTEGER,24),float_checks=360,retained_integer_checks=72,host_scalar_status_retained=True,actual_descriptors=dict.fromkeys(ORDER,{}))
  self.assertTrue(dispatch(d)['passed']);d['components']['gdn_prefill_core']['gdn_finite']['float_checks']=359
  with self.assertRaises(AssertionError):dispatch(d)
 def test_actual_prior_proofs_and_ast(self):
  self.assertTrue(qualification(R)['passed'])
  for f in R.glob('*.py'):ast.parse(f.read_text())
  self.assertFalse((R/'cache_v1').exists())
  for name in ('torch','vllm','cuda.tile'):self.assertNotIn(name,sys.modules)
if __name__=='__main__':unittest.main()
