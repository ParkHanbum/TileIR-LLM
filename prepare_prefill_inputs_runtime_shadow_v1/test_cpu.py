import unittest,ast,tempfile
from pathlib import Path
from unittest.mock import patch
from tileir_prepare_prefill_inputs_v1.contract import validate_values
from . import adapter
from .result_gate import native_finish_gate
R=Path(__file__).parent
class Tests(unittest.TestCase):
 def test_bounds(self):
  validate_values(list(range(32)),[i*53 for i in range(33)],[53]*32,[0]*32)
  for mapping,starts in [([0]*32,[i*53 for i in range(33)]),(list(range(32)),[i*100 for i in range(33)])]:
   with self.assertRaises(AssertionError):validate_values(mapping,starts,[53]*32,[0]*32)
 def test_skip_decode(self):validate_values(list(range(32)),[i*60 for i in range(33)],[53]*32,[53]*32)
 def test_latch(self):
  with tempfile.TemporaryDirectory() as d:
   a=adapter.Adapter(Path(d)/'o');a.start('shadow');calls=[]
   with patch.object(adapter,'validate',side_effect=ValueError('bad')):
    with self.assertRaises(ValueError):a.invoke(lambda *x:calls.append(1),())
    with self.assertRaises(AssertionError):a.invoke(lambda *x:calls.append(1),())
   self.assertEqual(calls,[])
 def test_ast(self):
  for root in [R,R.parent/'tileir_prepare_prefill_inputs_v1']:
   for p in root.glob('*.py'):ast.parse(p.read_text())
 def test_finish_gate(self):
  self.assertFalse(native_finish_gate({}));self.assertTrue(native_finish_gate(dict(shadow_only=True,replacement=False,failed=None,rows=[dict(passed=True,native_calls=1,tile_calls=1,outputs_raw=True,readonly_raw=True)])))

class Fake:
 def __init__(self,value):self.value=bytearray(value)
 def clone(self):return Fake(self.value)
class InvokeTests(unittest.TestCase):
 def run_case(self,wrong=False):
  with tempfile.TemporaryDirectory() as d:
   a=adapter.Adapter(Path(d)/'o');a.start('shadow');ts=tuple(Fake([i])for i in range(7));calls=[]
   def native(*xs):calls.append('native');xs[0].value[0]=91;xs[1].value[0]=92
   def tile(*xs):calls.append('tile');xs[0].value[0]=90 if wrong else 91;xs[1].value[0]=92
   with patch.object(adapter,'validate',lambda xs:None),patch.object(adapter,'raw',lambda t:bytes(t.value)),patch.object(adapter,'prepare_prefill_inputs',tile):
    if wrong:
     with self.assertRaises(AssertionError):a.invoke(native,ts)
     with self.assertRaises(AssertionError):a.invoke(native,ts)
     self.assertIsNotNone(a.failed)
    else:
     self.assertIsNone(a.invoke(native,ts));self.assertTrue(native_finish_gate(a.finish()));self.assertEqual(ts[0].value,bytearray([91]))
   self.assertEqual(calls,['native','tile'])
 def test_invoke_finish(self):self.run_case()
 def test_raw_failure_latched(self):self.run_case(True)
 def test_prerequisite(self):
  from .gates import qualification
  self.assertTrue(qualification(R)['independent_recovery'])
