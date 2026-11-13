import unittest,ast,json
from pathlib import Path
from .run import order,result_gate
from tileir_vllm_bool_not_v1.api import validate_contract
class T(unittest.TestCase):
 def test_contract(self):validate_contract((32,),(1,),'torch.bool',0,32)
 def test_reject(self):
  for a in [((31,),(1,),'torch.bool',0,32),((32,),(2,),'torch.bool',0,64),((32,),(1,),'torch.bool',1,33),((32,),(1,),'torch.int32',0,128),((32,),(1,),'torch.bool',0,32,True,False),((32,),(1,),'torch.bool',0,32,False,True)]:
   with self.assertRaises(AssertionError):validate_contract(*a)
 def test_logic(self):self.assertEqual([x==False for x in [False,True]],[True,False])
 def test_gate(self):
  keys=('raw_equal','input_owner_unchanged','layout','fresh_disjoint','earlier_outputs_unchanged','repeat_exact','same_current_stream','passed');rows=[dict(case=c,arm=a,repeat=n,**dict.fromkeys(keys,True))for c,a,n in order()];r=dict(complete=True,records=rows);self.assertTrue(result_gate(r)['raw_pass']);rows[0]['same_current_stream']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_actual(self):
  rows=json.loads(Path(__file__).with_name('OBSERVED.json').read_text());self.assertEqual(len(rows),24);self.assertTrue(all(x['values']==[False]*32 for x in rows))
 def test_ast(self):
  R=Path(__file__).parent
  for p in [*R.glob('*.py'),*(R.parent/'tileir_vllm_bool_not_v1').glob('*.py')]:ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
