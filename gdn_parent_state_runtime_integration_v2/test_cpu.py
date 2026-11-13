import unittest,tempfile,ast
from pathlib import Path
from .adapter import extent,values_contract,advance,Adapter,ORDER
from .result_gate import audit
R=Path(__file__).parent
class Tests(unittest.TestCase):
 def test_extent_gap_and_offset(self):
  e=extent((310,32,128,128),(540672,16384,128,1),128,4,1024**3);self.assertGreater(e['storage_extent_bytes'],e['logical_bytes']);self.assertEqual(e['storage_offset_bytes'],512)
 def test_extent_rejects_before_copy(self):
  for shape,stride,storage in [((10000,32,128,128),(540672,16384,128,1),64*1024**3),((310,32,128,128),(540672,16384,128,1),1024)]:
   with self.assertRaises(AssertionError):extent(shape,stride,0,4,storage)
 def test_indices_bounds_duplicates_mixedmask(self):
  a=values_contract(list(range(32)),[i%2==0 for i in range(32)],310);self.assertTrue(a['unique'])
  a=values_contract([-1]*32,[False]*32,310);self.assertFalse(a['unique']);self.assertEqual(a['normalized_indices'],[309]*32)
  with self.assertRaises(AssertionError):values_contract([310]*32,[False]*32,310)
 def test_exact_order_first_failure(self):
  row={'order':[]}
  for x in ORDER:advance(row,x)
  with self.assertRaises(AssertionError):advance(row,'gather')
  with self.assertRaises(AssertionError):advance({'order':[]},'scatter')
 def test_latch(self):
  with tempfile.TemporaryDirectory()as d:
   a=Adapter(Path(d)/'out');a.fail(ValueError('first'));f=a.failed.copy();a.fail(ValueError('second'));self.assertEqual(a.failed,f)
   with self.assertRaises(AssertionError):a.start('shadow')
 def test_no_owner_copy_or_numeric_port(self):
  s=(R/'adapter.py').read_text();self.assertNotIn('owner.clone',s);self.assertNotIn('owner.cpu',s);self.assertNotIn('ct.launch',s);self.assertIn('t.numel()*t.element_size()<=256',s)
 def test_ast_and_gate(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
  self.assertFalse(audit({})['passed'])

 def test_actual_finish_to_runner_receipt(self):
  import copy
  from .result_gate import native_finish_gate
  with tempfile.TemporaryDirectory()as d:
   a=Adapter(Path(d)/'out');a.start('shadow')
   a.rows=[dict(layer='layer'+str(i),passed=True,order=list(ORDER),gather_raw=True,mask_raw=True,scatter_raw=True,final_input_hash_preserved=True,unselected_padding_hash_preserved=True,selected_restore_hash_preserved=True,native_oracle_calls=3,tile_calls=3)for i in range(24)]
   receipt={'selected_dispatch':a.finish()}
   self.assertNotIn('shadow_pass',receipt['selected_dispatch']);self.assertTrue(native_finish_gate(receipt['selected_dispatch']))
   a.start('replace');a.rows=[dict(layer='layer'+str(i),passed=True,order=list(ORDER),gather_raw=True,mask_raw=True,scatter_raw=True,final_input_hash_preserved=True,unselected_padding_hash_preserved=True,selected_restore_hash_preserved=True,native_oracle_calls=3,tile_calls=3)for i in range(24)]
   self.assertTrue(native_finish_gate(a.finish()))
   bad=copy.deepcopy(receipt['selected_dispatch']);bad['rows'][0]['order'].reverse();self.assertFalse(native_finish_gate(bad))
   bad=copy.deepcopy(receipt['selected_dispatch']);bad['failed']={'error':'first'};self.assertFalse(native_finish_gate(bad))
   bad=copy.deepcopy(receipt['selected_dispatch']);bad['rows'].pop();self.assertFalse(native_finish_gate(bad))
 def test_runner_consumes_native_contract(self):
  s=(R/'run.py').read_text();self.assertIn("native_finish_gate(result['shadow_request']['selected_dispatch'])",s);self.assertNotIn("['selected_dispatch']['shadow_pass']",s)

class ChunkTests(unittest.TestCase):
 def test_complement_all_bytes(self):
  from .memory import complements,selected_intervals
  for indices in ([0,3],[3,0],[1],[]):
   regions=selected_intervals(indices,7,5);actual=[]
   for start in range(0,100,13):
    for lo,hi in complements(start,min(start+13,100),regions):actual.extend(range(lo,hi))
   expected=[i for i in range(100) if not any(a<=i<b for a,b in regions)]
   self.assertEqual(actual,expected)
 def test_failed_finish_cannot_enable_replace(self):
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'out');a.start('shadow');a.rows=[dict(layer=str(i),passed=True,order=list(ORDER)) for i in range(24)]
   with self.assertRaises(AssertionError):a.finish()
   self.assertIsNotNone(a.failed);self.assertFalse(a.baseline_pass)
   with self.assertRaises(AssertionError):a.start('replace')
 def test_scope_and_no_full_copy(self):
  s=(R/'adapter.py').read_text();m=(R/'memory.py').read_text()
  self.assertIn('native=func(*args,**kwargs);candidate=gather_state',s)
  self.assertIn('owner.index_copy_',s);self.assertIn('outside_view_verified=False',m)
  self.assertNotIn('owner.clone()',s);self.assertNotIn('owner.cpu()',s)

class PrerequisiteTests(unittest.TestCase):
 def receipts(self):
  import json
  p=R.parent/'gdn_parent_state_synthetic_v2'
  return [json.loads((p/f).read_text()) for f in ('gdnParentStateSynthetic02/RESULT.json','gdnParentStateSynthetic02-launch.json','gdnParentStateSynthetic02-recovery.json')]
 def test_actual_prerequisite(self):
  from .gates import qualification
  self.assertTrue(qualification(R)['actual_exit0'])
 def test_lifecycle_negative(self):
  from .gates import validate_receipts
  for pos,key,value in [(1,'returncode',None),(1,'returncode',True),(1,'returncode',1),(2,'actual_returncode',None),(2,'actual_returncode',False),(2,'actual_returncode',1),(1,'child_absent',False),(2,'child_absent',False),(1,'remaining_process_group',[1]),(2,'remaining_process_group',[1]),(1,'lock_reacquired',False),(2,'lock_reacquired',False),(2,'launch_sha256','bad'),(2,'source_pin_file_sha256','bad'),(2,'source_pins_verified',0),(1,'finished',None)]:
   args=self.receipts();args[pos][key]=value
   with self.subTest(key=key,pos=pos,value=value),self.assertRaises(AssertionError):validate_receipts(*args)
