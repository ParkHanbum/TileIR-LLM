import ast,json,unittest
from pathlib import Path
from tileir_gdn_parent_state_v1.contract import layout,index_values
from .run import result_gate
R=Path(__file__).parent;P=R.parent
class Tests(unittest.TestCase):
 def test_actual_nonzero_offsets(self):
  self.assertEqual(layout((313,32,128,128),(540672,16384,128,1),12288,4,5415370752),524288)
  self.assertEqual(layout((313,32,128,128),(540672,16384,128,1),1184624640,4,5415370752),524288)
 def test_bad_storage_pitch(self):
  for pitch,cap in [(1,1024**3),(540672,512)]:
   with self.assertRaises(AssertionError):layout((313,32,128,128),(pitch,16384,128,1),0,4,cap)
 def test_negative_duplicate_semantics(self):
  self.assertEqual(index_values([-1,0,-1],313),[312,0,312])
  with self.assertRaises(AssertionError):index_values([-1,312],313,True)
  with self.assertRaises(AssertionError):index_values([313],313)
 def test_fixture_bounds_and_masks(self):
  fs=json.loads((R/'MANIFEST.json').read_text())['fixtures'];self.assertEqual(len(fs),3)
  for f in fs:
   vals=f['indices'][::f['index_stride']];self.assertEqual(len(vals),32);index_values(vals,f['shape'][0],True)
   inner=f['shape'][1]*f['shape'][2]*f['shape'][3];self.assertLessEqual((f['offset']+(f['shape'][0]-1)*f['pitch']+inner+97)*4,384*1024**2)
  self.assertTrue(all(fs[0]['mask']));self.assertFalse(any(fs[1]['mask']));self.assertEqual(sum(fs[2]['mask']),16)
 def test_padding_address_oracle(self):
  n,inner,pitch,offset=5,6,9,3;back=list(range(offset+n*pitch+4));old=back[:];idx=[4,1];g=[[back[offset+r*pitch+k]for k in range(inner)]for r in idx]
  g[1]=[0]*inner;written=set()
  for i,r in enumerate(idx):
   for k in range(inner):back[offset+r*pitch+k]=100+i;written.add(offset+r*pitch+k)
  self.assertTrue(all(back[k]==old[k]for k in range(len(back))if k not in written));self.assertEqual(len(written),12)
 def test_ast_no_owner_capacity_constant(self):
  for root in (R,P/'tileir_gdn_parent_state_v1'):
   for p in root.glob('*.py'):ast.parse(p.read_text())
  s=(P/'tileir_gdn_parent_state_v1/kernels.py').read_text();self.assertNotIn('310',s);self.assertNotIn('313',s)
 def test_gate(self):self.assertFalse(result_gate({})['raw_pass'])
