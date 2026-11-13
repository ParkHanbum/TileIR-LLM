import ast,copy,json,unittest
from pathlib import Path
from unittest.mock import patch
from . import gates,cleanup
R=Path(__file__).parent
class Tests(unittest.TestCase):
 def fixture(self):
  ft,fp,opt=gates.frozen_reference()
  r=dict(complete=True,candidate_checks_passed=True,engine_options=opt,synthetic=dict(patterns=8,rows=224,phases=2,complete=True),observer_restored=True,transport_drained=True,engine_shutdown=True,synchronized_after_restore=True,metadata_progress=dict(pending_owners=0,failed=False),precision_policy=dict(initial=dict(bf16=True,split_k=True),restored=True,restored_split_k=True),cache_preseed=dict(destination=str(R/'cache_v1'),files_verified=True,separate_inodes_verified=True))
  for m in ('native','shadow','replace'):r[m+'_request']=dict(output_token_ids=ft,public_logprobs=fp,allow_bf16_reduced_precision_reduction=True,allow_bf16_reduced_precision_reduction_split_k=True)
  for m in ('shadow','replace'):
   t=dict(transfer_attempts=1,transfer_returns=1,cast_launch_attempts=1,cast_launch_returns=1,temporary_bytes=1792,source_pinned=True,same_dtype_transfer=True)
   row=dict(mode=m,raw=True,source_readonly=True,outside_preserved=True,native_original_calls=int(m=='shadow'),native_private_oracles=int(m=='replace'),candidate_calls=1,transport=t)
   r[m+'_census']=dict(mode=m,calls=1,rows=[row])
  return r
 def test_good(self):self.assertTrue(gates.audit(self.fixture())['passed'])
 def test_native_production_rejected(self):
  r=self.fixture();r['replace_census']['rows'][0]['native_original_calls']=1;self.assertFalse(gates.audit(r)['passed'])
 def test_transport_contract_rejected(self):
  for field,value in [('temporary_bytes',896),('same_dtype_transfer',False),('source_pinned',False),('cast_launch_returns',0)]:
   r=self.fixture();r['replace_census']['rows'][0]['transport'][field]=value;self.assertFalse(gates.audit(r)['passed'])
 def test_wrong_bytes_rejected(self):
  for field in ('raw','source_readonly','outside_preserved'):
   r=self.fixture();r['replace_census']['rows'][0][field]=False;self.assertFalse(gates.audit(r)['passed'])
 def test_pending_refs_rejected(self):
  r=self.fixture();r['metadata_progress']['pending_owners']=1;self.assertFalse(gates.audit(r)['passed'])
 def test_cleanup_restore_failure_still_drains(self):
  calls=[]
  class B:
   def close(self):calls.append('restore');raise RuntimeError()
  class T:
   def synchronize_and_release(self):calls.append('drain')
  class O:binding=B();transport=T()
  r={};cleanup.cleanup(r,lambda:None,None,None,O(),lambda:calls.append('sync'),lambda:None,True)
  self.assertEqual(calls,['restore','drain']);self.assertFalse(r['complete']);self.assertTrue(r['transport_drained'])
 def test_source_ast_and_order(self):
  for f in R.glob('*.py'):ast.parse(f.read_text())
  s=(R/'run.py').read_text();self.assertLess(s.index("result['synthetic']=synthetic(torch)"),s.index("native=request('native')"));self.assertLess(s.index("native=request('native')"),s.index('observer.binding.install()'))
  self.assertIn("('shadow','replace')",s);self.assertIn('observer.transport.ledger',s)
 def test_no_framework_import_on_module_load(self):
  import sys
  self.assertNotIn('torch',sys.modules);self.assertNotIn('vllm',sys.modules);self.assertNotIn('cuda.tile',sys.modules)
 def test_original_one_expression(self):
  from batch_ptr_int_cast_v1.binding import OLD
  self.assertEqual(OLD,ast.dump(ast.parse('batch_ptr[0:mlist_len].copy_(mlist, non_blocking=True)',mode='eval').body,include_attributes=False))
if __name__=='__main__':unittest.main()
