import unittest,struct,tempfile,ast
from pathlib import Path
from unittest.mock import patch
from . import adapter
from .result_gate import native_finish_gate
R=Path(__file__).parent
class Fake:
 def __init__(self,b):self.b=b
 def data_ptr(self):return id(self)
 def untyped_storage(self):return self
 def storage_offset(self):return 0
class Tests(unittest.TestCase):
 def args(self):
  pack=lambda xs,c='i':Fake(struct.pack('<'+c*len(xs),*xs))
  return (pack([-9]*2048),pack([-8]*32),pack(list(range(32)),'q'),pack([53*i for i in range(33)]),pack(list(range(128))*32),pack([53]*32),pack([0]*32))
 def test_invoke_original_addresses_and_counts(self):
  with tempfile.TemporaryDirectory()as d:
   a=adapter.Adapter(Path(d)/'out');calls=[]
   def work(*ts):
    ts[0].b=struct.pack('<'+'i'*2048,*(list(range(53))*32+[-9]*(2048-1696)));ts[1].b=bytes(128)
   def native(*ts):calls.append('native');work(*ts)
   def tile(*ts):calls.append('tile');work(*ts)
   with patch.object(adapter,'validate',lambda ts:None),patch.object(adapter,'raw',lambda t:t.b),patch.object(adapter,'prepare_prefill_inputs',tile):
    for mode in ('shadow','replace'):
     a.start(mode);ts=self.args();ids=[id(t)for t in ts];self.assertIsNone(a.invoke(native,ts));self.assertEqual(ids,[id(t)for t in ts]);self.assertTrue(native_finish_gate(a.finish()))
   self.assertEqual(calls,['native','tile'])
 def test_wrong_output_latches(self):
  with tempfile.TemporaryDirectory()as d:
   a=adapter.Adapter(Path(d)/'out');a.mode='replace';calls=[]
   with patch.object(adapter,'validate',lambda ts:None),patch.object(adapter,'raw',lambda t:t.b),patch.object(adapter,'prepare_prefill_inputs',lambda *ts:calls.append(1)):
    with self.assertRaises(AssertionError):a.invoke(lambda *ts:None,self.args())
    with self.assertRaises(AssertionError):a.invoke(lambda *ts:None,self.args())
   self.assertEqual(calls,[1]);self.assertIsNotNone(a.failed)
 def test_prerequisite(self):
  from .gates import qualification
  self.assertTrue(qualification(R)['actual_shadow_pass'])
 def test_ast(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
