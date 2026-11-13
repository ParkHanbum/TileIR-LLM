import unittest,ast,copy
from pathlib import Path
from .result_gate import exact_public_gate
class Tests(unittest.TestCase):
 def rows(self):return [[{str(k):dict(logprob=-float(k+1),rank=k+1) for k in range(5)} for _ in range(16)] for _ in range(32)]
 def test_actual_sampling_flag(self):
  t=ast.parse(Path(__file__).with_name('run.py').read_text());calls=[n for n in ast.walk(t) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='SamplingParams'];self.assertEqual(len(calls),1)
  self.assertEqual(next(x.value.value for x in calls[0].keywords if x.arg=='logprobs'),5)
 def test_exact(self):
  a=self.rows();self.assertEqual(exact_public_gate(a,copy.deepcopy(a))['values'],2560)
 def test_small_difference_is_not_allowed(self):
  a=self.rows();b=copy.deepcopy(a);b[0][0]['0']['logprob']+=1e-7
  with self.assertRaises(AssertionError):exact_public_gate(a,b)
 def test_rank_and_count(self):
  a=self.rows();b=copy.deepcopy(a);b[0][0]['0']['rank']=2
  with self.assertRaises(AssertionError):exact_public_gate(a,b)
  for row in a:
   for step in row:step.pop('4')
  with self.assertRaises(AssertionError):exact_public_gate(a,a)
