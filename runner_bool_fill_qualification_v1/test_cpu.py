import unittest,json,ast,sys,builtins
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS
from .run import result_gate,order
from tileir_vllm_runner_bool_fill_v1.api import fill_padding_
class T(unittest.TestCase):
 def test_gate(self):
  rows=[dict(case=c,arm=a,repeat=n,raw_equal=True,outside_unchanged=True,return_identity=True,layout=True,repeat_exact=True,passed=True)for c,a,n in order()];r=dict(complete=True,records=rows);self.assertTrue(result_gate(r)['raw_pass']);rows[0]['outside_unchanged']=False;self.assertFalse(result_gate(r)['raw_pass'])
 def test_order(self):
  self.assertEqual(len(order()),24);self.assertEqual(len(set(order())),24)
 def test_empty_no_cuda_import(self):
  class V:
   shape=(0,);dtype='torch.bool';device=NS(type='cuda');requires_grad=False
   def stride(self):return (1,)
   def storage_offset(self):return 13
   def untyped_storage(self):return NS(nbytes=lambda:32)
  v=V();orig=builtins.__import__
  def guard(name,*args,**kw):
   if name.startswith('cuda'):raise AssertionError('Empty must not import or launch CUDA Tile')
   return orig(name,*args,**kw)
  with patch.dict(sys.modules,{'torch':NS()}),patch('builtins.__import__',guard):self.assertIs(fill_padding_(v,True),v)
 def test_manifest(self):
  m=json.loads(Path(__file__).with_name('MANIFEST.json').read_text());self.assertEqual([c['name']for c in m['cases']],[x[0]for x in order()[::4]])
 def test_ast(self):
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
