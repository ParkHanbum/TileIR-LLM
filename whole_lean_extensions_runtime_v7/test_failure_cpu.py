import unittest,json
from types import SimpleNamespace as N
from .failure import record,snapshot
class Tests(unittest.TestCase):
 def test_closed_labels(self):
  e=RuntimeError('0xDEADBEEF tensor([private])')
  r=record('0xDEADBEEF',e)
  self.assertEqual(r,dict(component='other',error_type='RuntimeError',out_of_memory=False))
  self.assertNotIn('DEADBEEF',json.dumps(r))
 def test_primary_not_replaced_by_outer_latch(self):
  primary=record('gdn_prefill_core',MemoryError('private'))
  ext=N(controller=N(primary_failure=primary),bus=N(failure=dict(component='output_copy/entry',error_type='AssertionError')),output_copy=N(failed=True),ownership_pair=N(failed=False))
  r=snapshot(ext);self.assertEqual(r['controller_primary'],primary);self.assertTrue(r['controller_primary']['out_of_memory'])
 def test_unknown_exception_name_not_disclosed(self):
  Evil=type('Secret_Name',(Exception,),{})
  self.assertEqual(record('all_gemm',Evil())['error_type'],'OtherException')
