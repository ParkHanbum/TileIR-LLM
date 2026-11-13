import unittest,ast,json,hashlib,textwrap,inspect
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from runner_constructors_runtime_integration_v1 import adapter
from tileir_vllm_runner_constructors_v1 import api
R=Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def setUp(self):
  self.guard=patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CUDA forbidden'));self.g=self.guard.start()
  self.a=adapter.Adapter('/tmp/runner-unused-failure');self.a.original=lambda runner,scheduler_output,batch_req_state,batch_desc,num_active_loras:None
  self.runner=NS(model_state=NS(num_new_sampled_tokens_per_step=1),pcp_manager=None)
  self.s=NS(scheduled_spec_decode_tokens={});self.b=NS(req_ids=list(range(32)))
 def tearDown(self):self.g.assert_not_called();self.guard.stop()
 def copied(self,r,*args):
  x=self.a.construct('cu_num_logits',32,'cpu');y=self.a.construct('expanded_local_pos',32,'cpu');mapping=object()
  return NS(cu_num_logits=x,expanded_local_pos=y,req_ids=self.b.req_ids,idx_mapping=mapping,expanded_idx_mapping=mapping,num_reqs=32,num_draft_tokens=0)
 def invoke(self):return self.a.invoke(self.copied,self.runner,self.s,self.b,None,0)
 def test_modes(self):
  with patch.object(api,'cu_num_logits',lambda n,d:torch.arange(33,dtype=torch.int32)),patch.object(api,'expanded_local_pos',lambda n,d:torch.zeros(32,dtype=torch.int32)):
   for mode,native in [('shadow',32),('replace',0)]:
    self.a.start(mode)
    for i in range(16):self.invoke()
    d=self.a.finish();self.assertEqual(d['native_constructor_calls'],native);self.assertEqual(d['tileir_constructor_calls'],32)
 def test_exception_latch(self):
  self.a.start('shadow')
  with patch.object(api,'cu_num_logits',side_effect=RuntimeError('mock failure')) as f:
   with self.assertRaises(RuntimeError):self.invoke()
   with self.assertRaises(AssertionError):self.invoke()
   self.assertEqual(f.call_count,1)
  self.assertTrue(self.a.failed)
 def test_unsupported_before_body(self):
  self.a.start('shadow');self.s.scheduled_spec_decode_tokens={'a':[1]}
  with self.assertRaises(AssertionError):self.invoke()
  self.assertEqual(self.a.rows,[])
 def test_no_replace_before_shadow(self):
  with self.assertRaises(AssertionError):self.a.start('replace')
 def test_exact_copy_delta(self):
  o=(R/'prepare_inputs.original.py.txt').read_text();c=(R/'prepare_inputs.copied.py.txt').read_text()
  c=c.replace("_construct('cu_num_logits', num_reqs, self.device)",'torch.arange(\n            num_reqs + 1, device=self.device, dtype=torch.int32\n        )').replace("_construct('expanded_local_pos', num_reqs, self.device)",'torch.zeros(\n            num_reqs, dtype=torch.int32, device=self.device\n        )')
  self.assertEqual(c,o);ast.parse(c)
 def test_source_and_files(self):
  p=json.loads((R/'SOURCE.json').read_text());self.assertEqual(hashlib.sha256(Path(p['path']).read_bytes()).hexdigest(),p['sha256'])
  for f in R.glob('*.py'):ast.parse(f.read_text())
  self.assertFalse((R/'cache_v1').exists());self.assertFalse((R/'CACHE-PRESEED-01.json').exists())
 def test_partial_count(self):
  self.a.start('shadow')
  with self.assertRaises(AssertionError):self.a.finish()
if __name__=='__main__':unittest.main()
