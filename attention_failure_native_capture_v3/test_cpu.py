import unittest,ast,tempfile,json,functools
from pathlib import Path
from .observer import select_case,Observer,CAP
from .result_gate import audit
class Tests(unittest.TestCase):
 def test_selection(self):
  rows=[(f'language_model.model.layers.{i}.self_attn.attn',L) for L in range(53,69) for i in (3,7,11,15,19,23,27,31)]
  chosen=[(j,select_case(j,n,[L]*32)) for j,(n,L) in enumerate(rows) if select_case(j,n,[L]*32)]
  self.assertEqual(chosen,[(1,'second_attention_call'),(96,'late_65'),(120,'late_68')])
 def test_second_identity_not_hardcoded(self):self.assertEqual(select_case(1,'unexpected_observed_owner',[53]*32),'second_attention_call')
 def test_latch_context_preserved(self):
  with tempfile.TemporaryDirectory() as t:
   o=Observer(Path(t)/'bank');o.pending=dict(name='actual_module',phase='prefill',starts=[0,53],lens=[53],calls=1)
   def bad():raise AssertionError('first')
   f=o.latch(bad)
   with self.assertRaises(AssertionError):f()
   o.pending=None
   with self.assertRaises(AssertionError):o.latch(bad)()
   self.assertEqual(o.first_failure['module'],'actual_module');self.assertIn('first',o.first_failure['error'])
 def test_exact_once_identity_wrapper(self):
  with tempfile.TemporaryDirectory() as t:
   o=Observer(Path(t)/'bank');calls=[];obj=object()
   def original():calls.append(1);return obj
   self.assertIs(o.latch(original)(),obj);self.assertEqual(calls,[1])
 def test_gate_negative(self):self.assertFalse(audit({})['passed'])
 def test_no_candidate(self):
  s=Path(__file__).with_name('observer.py').read_text();self.assertNotIn('tileir_paged_attention',s);self.assertNotIn('ct.launch',s)
  self.assertIn('prior_q_logical_match',s);self.assertIn('prior_native_logical_match',s)
 def test_ast_bounds(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
  self.assertEqual(CAP,256*1024*1024)
if __name__=='__main__':unittest.main()
