import unittest,ast,json,struct
from pathlib import Path
from .run import validate_manifest,result_gate
from tileir_paged_attention_v10.contract import validate_values
R=Path(__file__).parent;P=R.parent
class Tests(unittest.TestCase):
 def test_contract_lengths(self):
  for L in range(54,69):self.assertEqual(validate_values(list(range(33)),[L]*32,[[0]*8]*32,310,528),1)
  for L in (52,69):
   with self.assertRaises(AssertionError):validate_values(list(range(33)),[L]*32,[[0]*8]*32,310,528)
  with self.assertRaises(AssertionError):validate_values([53*i for i in range(33)],[65]*32,[[0]*8]*32,310,528)
 def test_mapping(self):
  cols=[[8*n+2*l+j for n in range(8) for j in range(2)] for l in range(4)]
  self.assertEqual(sorted(sum(cols,[])),list(range(64)))
  for L in range(65,69):
   visits=[64+c for lane in cols for c in lane if 64+c<L]+[c for lane in cols for c in lane]
   self.assertEqual(sorted(visits),list(range(L)))
 def test_recurrence(self):
  # Integer-exact positive inputs: validates carry location, not GPU FP32 equivalence.
  cols=[[8*n+2*l+j for n in range(8) for j in range(2)] for l in range(4)]
  tail=[1 if c<4 else 0 for c in range(64)];base=list(range(1,65));alpha=2
  sums=[sum(tail[c] for c in lane) for lane in cols]
  for l,lane in enumerate(cols):
   sums[l]*=alpha
   for c in lane:sums[l]+=base[c]
  self.assertEqual((sums[0]+sums[2])+(sums[1]+sums[3]),sum(tail)*alpha+sum(base))
 def test_dispatch_unchanged(self):
  s=(P/'tileir_paged_attention_v10/api.py').read_text();self.assertIn('from tileir_paged_attention_v9.kernels import kernel',s);self.assertIn('width==1 and int(lens[0].item())>64',s)
 def test_direct_carry(self):
  s=(P/'tileir_paged_attention_v10/kernels.py').read_text();self.assertLess(s.index('pos=col+64'),s.index('pos=col+0'));self.assertIn('pv=ct.mma(pm,vv,pv)',s);self.assertEqual(s.count('s02=ct.add'),1);self.assertEqual(s.count('sums=ct.add'),31);self.assertIn('sums=ct.mul(sums,alpha',s)
 def test_bank_order(self):
  ss=validate_manifest(json.loads((R/'MANIFEST.json').read_text()));self.assertEqual(len(ss),6);self.assertEqual([x['lengths'][0] for x in ss[:2]],[65,68]);self.assertIn('.layers.11.',ss[2]['module'])
 def test_gate(self):
  r=dict(complete=True,records=[dict(passed=True) for _ in range(24)]);self.assertTrue(result_gate(r)['raw_pass']);r['records'][7]['passed']=False;self.assertFalse(result_gate(r)['raw_pass']);self.assertFalse(result_gate(dict(complete=True,records=[]))['raw_pass'])
 def test_failstop(self):
  s=(R/'run.py').read_text();self.assertIn("('native','native','tile','tile')",s);self.assertIn("r['raw_difference_hold']=True;save();return 2",s)
 def test_ast(self):
  for d in (R,P/'tileir_paged_attention_v10'):
   for f in d.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
