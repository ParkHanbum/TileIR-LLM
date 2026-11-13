import unittest,ast,json,hashlib
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from runner_hidden_gather_runtime_integration_v1 import adapter
import tileir_vllm_hidden_gather_v1 as api
R=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def setUp(self):
  self.guard=patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA forbidden'));self.mock=self.guard.start()
  self.a=adapter.Adapter('/tmp/unused-hidden-failure');self.a.model=object();self.runner=NS(model=self.a.model,batch_sharder=None);self.marker=object();self.calls=0
 def tearDown(self):self.mock.assert_not_called();self.guard.stop()
 def copied(self,r,x,idx):
  self.calls+=1;self.a.gather(x,idx);return self.marker
 def test_modes_return_identity_counts(self):
  with patch.object(api,'gather_hidden_rows',lambda x,i:x[i]):
   for mode,native in [('shadow',16),('replace',0)]:
    self.a.start(mode)
    for i in range(16):
     x=torch.zeros((1696 if i==0 else 32,2560),dtype=torch.bfloat16);idx=torch.arange(32)
     self.assertIs(self.a.invoke(self.copied,self.runner,x,idx),self.marker)
    self.assertEqual(self.a.finish()['native_gather_calls'],native)
  self.assertEqual(self.calls,32)
 def test_latch(self):
  self.a.start('shadow')
  with patch.object(api,'gather_hidden_rows',side_effect=RuntimeError('mock')) as f:
   for exc in (RuntimeError,AssertionError):
    with self.assertRaises(exc):self.a.invoke(self.copied,self.runner,torch.zeros(1696,2560,dtype=torch.bfloat16),torch.arange(32))
   self.assertEqual(f.call_count,1)
 def test_sharding_before_method(self):
  self.a.start('shadow');self.runner.batch_sharder=object()
  with self.assertRaises(AssertionError):self.a.invoke(self.copied,self.runner,None,None)
  self.assertEqual(self.calls,0)
 def test_no_early_replacement(self):
  with self.assertRaises(AssertionError):self.a.start('replace')
 def test_exact_copy_and_source(self):
  o=(R/'sample.original.py.txt').read_text();c=(R/'sample.copied.py.txt').read_text();self.assertEqual(c.replace('_gather(hidden_states, input_batch.logits_indices)','hidden_states[input_batch.logits_indices]'),o)
  p=json.loads((R/'SOURCE.json').read_text());self.assertEqual(hashlib.sha256(Path(p['path']).read_bytes()).hexdigest(),p['sha256'])
  for f in R.glob('*.py'):ast.parse(f.read_text())
 def test_gate_failure(self):
  from runner_hidden_gather_runtime_integration_v1.result_gate import audit
  self.assertFalse(audit({})['passed'])
if __name__=='__main__':unittest.main()
