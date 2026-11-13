import unittest,json,ast,sys,types
from pathlib import Path
from unittest.mock import patch
from .predicate import finite_boolean
R=Path(__file__).parent
class T(unittest.TestCase):
 def test_boolean_error_semantics(self):
  self.assertTrue(finite_boolean(None,lambda x:None))
  def bad(x):raise AssertionError('NaN/Inf outside finite-only contract')
  self.assertFalse(finite_boolean(None,bad))
  def other(x):raise AssertionError('Descriptor failed')
  with self.assertRaisesRegex(AssertionError,'Descriptor failed'):finite_boolean(None,other)
 def test_predicate_only_deltas(self):
  m=json.loads((R/'MANIFEST.json').read_text());self.assertEqual(sum(len(x['sites'])for x in m['bodies']),6)
  for x in m['bodies']:
   s=(R/'bodies'/(x['name']+'.original.py.txt')).read_text()
   for sid,expr in x['replacements']:
    var=expr[len('bool(torch.isfinite('):].split(')')[0];s=s.replace(expr,f'_finite_all({sid!r},{var})')
   self.assertEqual(s,(R/'bodies'/(x['name']+'.candidate.py.txt')).read_text());ast.parse(s)
 def test_scores_actual_body_shortcircuit(self):
  touched=[]
  class Tensor:
   shape=(32,248320);dtype='torch.bfloat16';requires_grad=False;device=types.SimpleNamespace(type='cuda')
   def stride(self):return (248320,1)
   def storage_offset(self):return 0
   def __ge__(self,x):touched.append('range');raise AssertionError('range must not run')
  torch=types.SimpleNamespace(Tensor=Tensor,empty=lambda *a,**k:touched.append('allocation'));tile=types.SimpleNamespace(launch=lambda *a:touched.append('launch'))
  ns=dict(__package__='tileir_vllm_selected_logprobs_v2',validate_descriptors=lambda *a:None,_finite_all=lambda *a:False)
  exec((R/'bodies/scores_api.candidate.py.txt').read_text(),ns)
  with patch.dict(sys.modules,{'torch':torch,'cuda':types.SimpleNamespace(tile=tile),'cuda.tile':tile,'tileir_vllm_selected_logprobs_v2.kernels':types.SimpleNamespace(kernel=None)}):
   with self.assertRaises(AssertionError)as e:ns['compute_token_logprobs'](Tensor(),Tensor())
  self.assertEqual(str(e.exception),'');self.assertEqual(touched,[])
 def test_copy_actual_body_valueerror_before_cuda(self):
  class Tensor:
   shape=(32,248320);dtype='torch.bfloat16';requires_grad=False;layout='strided';device=types.SimpleNamespace(type='cuda')
   def stride(self):return (248320,1)
   def storage_offset(self):return 0
   def is_conj(self):return False
   def is_neg(self):return False
  ns=dict(validate_metadata=lambda m:m['shape'],_finite_all=lambda *a:False);exec((R/'bodies/copy_api.candidate.py.txt').read_text(),ns)
  with patch.dict(sys.modules,{'torch':types.SimpleNamespace(Tensor=Tensor,strided='strided')}):
   with self.assertRaisesRegex(ValueError,'nonfinite payloads not qualified; no launch'):ns['sampler_fp32_copy'](Tensor())
 def test_postcheck_order(self):
  s=(R/'bodies/hidden_post.candidate.py.txt').read_text();self.assertLess(s.index('y=self.call'),s.index("_finite_all('hidden_post'"));self.assertLess(s.index("_finite_all('hidden_post'"),s.index('self.current=y'))
  s=(R/'bodies/sampler_outer.candidate.py.txt').read_text();self.assertLess(s.index("_finite_all('sampler_pre'"),s.index('y=self.call'));self.assertLess(s.index('y=self.call'),s.index("_finite_all('sampler_post'"))
 def test_greedy_excluded(self):
  m=json.loads((R/'MANIFEST.json').read_text());self.assertFalse(any('greedy'in x['name']for x in m['bodies']));rows=json.loads((R/'OBSERVED-EIGHT-SITES.json').read_text());self.assertEqual(len(rows),8);self.assertEqual(sum(x['calls']for x in rows),128)
 def test_no_cuda_import(self):self.assertNotIn('cuda.tile',sys.modules)
if __name__=='__main__':unittest.main()
