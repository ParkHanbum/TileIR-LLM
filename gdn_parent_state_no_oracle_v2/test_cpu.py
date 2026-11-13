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
   a.rows=[dict(layer='layer'+str(i),passed=True,order=list(ORDER),native_target_calls=3 if a.mode=='shadow' else 0,tile_target_calls=0 if a.mode=='shadow' else 3,native_invert_calls=1 if a.mode=='shadow' else 0,tile_invert_calls=0 if a.mode=='shadow' else 1)for i in range(24)]
   receipt={'selected_dispatch':a.finish()}
   self.assertNotIn('shadow_pass',receipt['selected_dispatch']);self.assertTrue(native_finish_gate(receipt['selected_dispatch']))
   a.start('replace');a.rows=[dict(layer='layer'+str(i),passed=True,order=list(ORDER),native_target_calls=3 if a.mode=='shadow' else 0,tile_target_calls=0 if a.mode=='shadow' else 3,native_invert_calls=1 if a.mode=='shadow' else 0,tile_invert_calls=0 if a.mode=='shadow' else 1)for i in range(24)]
   self.assertTrue(native_finish_gate(a.finish()))
   bad=copy.deepcopy(receipt['selected_dispatch']);bad['rows'][0]['order'].reverse();self.assertFalse(native_finish_gate(bad))
   bad=copy.deepcopy(receipt['selected_dispatch']);bad['failed']={'error':'first'};self.assertFalse(native_finish_gate(bad))
   bad=copy.deepcopy(receipt['selected_dispatch']);bad['rows'].pop();self.assertFalse(native_finish_gate(bad))
 def test_runner_consumes_native_contract(self):
  s=(R/'run.py').read_text();self.assertIn("native_finish_gate(result['shadow_request']['selected_dispatch'])",s);self.assertNotIn("['selected_dispatch']['shadow_pass']",s)

class NoOracleTests(unittest.TestCase):
 def test_pending_proof_blocks(self):
  from .gates import qualification
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(AssertionError):qualification(Path(d))
 def test_no_value_diagnostics(self):
  s=(R/'adapter.py').read_text()
  for forbidden in ('owner_hashes','same(','index_copy_','preselected','clone(','tensor_hash'):
   self.assertNotIn(forbidden,s)
  self.assertIn("if self.mode=='replace' and label in ('gather','invert','mask','scatter')",s)
 def test_restore_order_is_reverse(self):
  s=(R/'adapter.py').read_text();self.assertIn('reversed(self.bindings)',s)

class InvokeTests(unittest.TestCase):
 def test_actual_invoke_to_finish(self):
  from unittest.mock import patch
  from . import adapter
  from .result_gate import native_finish_gate
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'out')
   for mode in ('shadow','replace'):
    a.start(mode);calls=[]
    with patch.multiple(adapter,gather_state=lambda *x:calls.append('gather'),bitwise_not_bool32=lambda *x:calls.append('invert'),mask_initial_state_=lambda *x:calls.append('mask'),scatter_state_=lambda *x:calls.append('scatter')):
     for i in range(24):
      row=dict(layer=str(i),order=[],passed=False,native_target_calls=0,tile_target_calls=0,native_invert_calls=0,tile_invert_calls=0);a.rows.append(row)
      ctx=dict(row=row,owner=object(),indices=object(),mask=object(),initial=object(),final=object())
      for label in ORDER:
       advance(row,label)
       if label in ('gather','invert','mask','scatter'):a.invoke(label,lambda: calls.append('native'),(),{},ctx)
      row['passed']=True
    self.assertTrue(native_finish_gate(a.finish()))
    self.assertEqual(calls.count('native'),96 if mode=='shadow' else 0)
    if mode=='replace':self.assertEqual([calls.count(x)for x in ('gather','invert','mask','scatter')],[24]*4)
 def test_invoke_failure_latches_no_later_call(self):
  from unittest.mock import patch
  from . import adapter
  with tempfile.TemporaryDirectory() as d:
   a=Adapter(Path(d)/'out');a.mode='replace';row=dict(tile_invert_calls=0);ctx=dict(row=row,mask=object());calls=[]
   def bad(*x):calls.append('tile');raise RuntimeError('first')
   with patch.object(adapter,'bitwise_not_bool32',bad):
    with self.assertRaises(RuntimeError):a.invoke('invert',lambda:calls.append('native'),(),{},ctx)
    first=a.failed.copy()
    with self.assertRaises(AssertionError):a.invoke('invert',lambda:calls.append('native'),(),{},ctx)
   self.assertEqual(calls,['tile']);self.assertEqual(a.failed,first)
 def test_actual_prerequisites(self):
  from .gates import qualification
  self.assertTrue(qualification(R)['bool_not'])
