import unittest,tempfile,json,hashlib,ast
from pathlib import Path
from unittest.mock import patch
import torch
import tileir_vllm_sampler_fp32_copy_v1 as api
from sampler_fp32_copy_runtime_integration_v1.adapter import Adapter
R=Path(__file__).parent
class Tests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(4);self.guard=patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('NoCUDA'));self.g=self.guard.start()
 def tearDown(self):self.g.assert_not_called();self.guard.stop()
 def test_modes_and_processor_mutation(self):
  a=Adapter('/tmp/unused-cast-failure');owner=object();x=torch.zeros(32,248320,dtype=torch.bfloat16)
  def method(o,x):y=a.copy_logits(x);y.fill_(-float('inf'));return y
  with patch.object(api,'sampler_fp32_copy',lambda x:x.float()):
   for mode,n in [('shadow',16),('replace',0)]:
    a.start(mode)
    for i in range(16):self.assertIs(a.invoke(method,owner,x),a.current)
    d=a.finish();self.assertEqual(d['native_copy_calls'],n);self.assertTrue(all(z['post_processor_buffer_changed']and z['post_method_inputs_preserved']for z in d['rows']))
 def test_nonfinite_before_native_tile(self):
  a=Adapter('/tmp/unused-cast-failure');a.start('shadow');x=torch.full((32,248320),float('inf'),dtype=torch.bfloat16)
  with patch.object(torch,'empty_like',side_effect=AssertionError('native forbidden')) as native,patch.object(api,'sampler_fp32_copy') as tile:
   with self.assertRaises(AssertionError):a.invoke(lambda o,x:a.copy_logits(x),object(),x)
   self.assertEqual(native.call_count,0);self.assertEqual(tile.call_count,0)
  self.assertTrue(a.failed)
 def test_latch_and_early_replace(self):
  a=Adapter('/tmp/unused-cast-failure')
  with self.assertRaises(AssertionError):a.start('replace')
  a.start('shadow');calls=[]
  def bad(owner):calls.append(1);raise ValueError('native')
  with self.assertRaises(ValueError):a.invoke(bad,object())
  with self.assertRaises(AssertionError):a.invoke(bad,object())
  self.assertEqual(calls,[1])
 def test_copy_diff_source(self):
  old=(R/'method.original.py.txt').read_text();new=(R/'method.copied.py.txt').read_text();self.assertEqual(new.replace('_copy_logits(logits)','torch.empty_like(logits, dtype=torch.float32).copy_(logits)'),old)
  pin=json.loads((R/'SOURCE.json').read_text());self.assertEqual(hashlib.sha256(Path(pin['path']).read_bytes()).hexdigest(),pin['sha256'])
  for p in R.glob('*.py'):ast.parse(p.read_text())
 def test_gate_failclosed(self):
  from sampler_fp32_copy_runtime_integration_v1.result_gate import audit
  self.assertFalse(audit({})['passed'])
if __name__=='__main__':unittest.main()
