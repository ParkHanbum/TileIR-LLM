import unittest,math,ast,sys
from pathlib import Path
from .contract import descriptor,status_boolean,scalar_reference
R=Path(__file__).parent
class T(unittest.TestCase):
 def test_values(self):
  self.assertTrue(scalar_reference([0.,-0.,5e-324,1.7976931348623157e308]));self.assertTrue(scalar_reference([]))
  for x in (float('nan'),-float('nan'),float('inf'),-float('inf')):self.assertFalse(scalar_reference([0.,x]))
 def test_size_and_integer_sum(self):
  for n in (54272,3473408,6946816,16777216):
   self.assertEqual(descriptor((n,),'torch.float32',0,True),n);self.assertLess(n,2**31)
   blocks=(n+1023)//1024;self.assertLessEqual(1<<(blocks-1).bit_length(),16384)
  with self.assertRaises(AssertionError):descriptor((16777217,),'torch.float32',0,True)
  self.assertEqual(descriptor((0,),'torch.bfloat16',0,True),0);self.assertTrue(status_boolean(0,0))
 def test_exact_delta_and_order(self):
  old=(R/'apply.original.py.txt').read_text();new=(R/'apply.candidate.py.txt').read_text()
  self.assertEqual(new,old.replace('bool(torch.isfinite(t).all())','_gdn_finite(t)'))
  self.assertLess(new.index('assert all(_gdn_finite'),new.index("assert bool((inputs['initial_state']==0).all())"))
  self.assertIn('label not in retained and _gdn_finite(t) and t.is_contiguous()',new)
 def test_shortcircuit(self):
  seen=[]
  def pred(v):seen.append(v);return v!=2
  self.assertFalse(all(pred(x)for x in [1,2,3]));self.assertEqual(seen,[1,2])
  self.assertTrue(-0.0==0);self.assertFalse(float('nan')==0)
 def test_actual_input_assert_shortcircuit(self):
  text=(R/'apply.candidate.py.txt').read_text();tree=ast.parse(text)
  f=tree.body[0];checks=[x for x in f.body if isinstance(x,ast.Assert)]
  finite=next(x for x in checks if '_gdn_finite' in ast.unparse(x))
  zero=next(x for x in checks if "initial_state" in ast.unparse(x) and '== 0' in ast.unparse(x))
  seen=[]
  def pred(v):seen.append(v);return v!=2
  ns={'inputs':{'first':1,'second':2,'third':3},'_gdn_finite':pred}
  fragment=ast.Module(body=[finite,zero],type_ignores=[]);ast.fix_missing_locations(fragment)
  with self.assertRaises(AssertionError)as e:exec(compile(fragment,'actual-input-asserts','exec'),ns)
  self.assertEqual(str(e.exception),'');self.assertEqual(seen,[1,2]) # zero cannot access missing initial_state
 def test_actual_observe_postcheck_order(self):
  tree=ast.parse((R/'apply.candidate.py.txt').read_text());observe=next(n for n in tree.body[0].body if isinstance(n,ast.FunctionDef)and n.name=='observe')
  retained={};seen=[]
  ns={'retained':retained,'inputs':{},'_gdn_finite':lambda x:seen.append('finite')or False,'overlap':lambda *a:seen.append('overlap')}
  exec(compile(ast.Module(body=[observe],type_ignores=[]),'actual-observe','exec'),ns)
  class Produced:
   def is_contiguous(self):seen.append('layout');return True
  x=Produced();seen.append('producer')
  with self.assertRaises(AssertionError)as e:ns['observe']({'g_cumsum':x})
  self.assertEqual(str(e.exception),'');self.assertEqual(seen,['producer','finite']);self.assertEqual(retained,{})
 def test_last_indices_and_scratch(self):
  n=2**24;b=n//1024
  self.assertEqual((b-1)*1024+1023,n-1);self.assertEqual(b,16384)
  self.assertEqual(b*4,65536);self.assertLess((n-1)*4,2**31)
 def test_ast_only_no_gpu_import(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
  for x in ('torch','cuda.tile','vllm'):self.assertNotIn(x,sys.modules)
if __name__=='__main__':unittest.main()
