import ast,unittest
from pathlib import Path
from .contract import run_stages,reference,MESSAGES
from .cases import cases,arguments
class Tests(unittest.TestCase):
 def test_cases(self):
  for name,d,expected in cases():
   with self.subTest(name=name):
    if expected:
     with self.assertRaisesRegex(ValueError,'^'+expected+'$'):reference(*arguments(d))
    else:self.assertIsNone(reference(*arguments(d)))
 def test_fail_fast(self):
  for failing in range(9):
   visits=[]
   def evaluate(s):visits.append(s);return int(s==failing)
   with self.assertRaisesRegex(ValueError,'^'+MESSAGES[failing]+'$'):run_stages(evaluate)
   self.assertEqual(visits,list(range(failing+1)))
 def test_empty_all_stages(self):
  seen=[];run_stages(lambda s:seen.append(s)or 0);self.assertEqual(seen,list(range(9)))
 def test_status_bounds(self):
  for value in (-1,33,True,None):
   with self.assertRaises(RuntimeError):run_stages(lambda s:value)
 def test_dependent_hist_guard(self):
  d=cases()[14][1];d['sampled'][0]=-1
  args=list(arguments(d));args[2]=lambda *_:self.fail('Histogram accessed before bounds')
  with self.assertRaisesRegex(ValueError,'histogram/token bounds'):reference(*args)
 def test_source_no_framework_import(self):
  for name in ('api.py','binding.py','qualify.py','contract.py'):
   tree=ast.parse((Path(__file__).parent/name).read_text())
   imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
   self.assertFalse(any(getattr(n,'module','') in ('torch','cuda.tile')for n in imports))
 def test_kernel_integer_promotions(self):
  text=(Path(__file__).parent/'kernel.py').read_text();ast.parse(text)
  self.assertIn('ct.int64',text);self.assertNotIn('ct.int32)',text.split('elif stage==6:')[1].split('ct.scatter')[0])

class BindingTests(unittest.TestCase):
 def test_actual_source_copy_and_restore(self):
  import types,sys
  from unittest.mock import patch
  from .binding import SOURCE,Binding
  source=ast.parse(SOURCE.read_text());cls=next(n for n in source.body if isinstance(n,ast.ClassDef)and n.name=='Replacement');node=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='call')
  ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
  Owner=type('Owner',(),{'call':ns['call']});owner=Owner();owner.rows=[];seen=[]
  def callback(torch,args):seen.append(args);raise ValueError('Sentinel before numerical kernel')
  binding=Binding(owner,callback);binding.install()
  self.assertIs(owner.call.__func__.__globals__['_post_update_value_guard'],callback)
  fake_torch=types.ModuleType('torch');fake_torch.Tensor=type('FakeTensor',(),{})
  fake_contract=types.ModuleType('tileir_vllm_post_update_v3.contract');fake_contract.NAMES=tuple('abcdefghij');fake_contract.validate_metadata=lambda args:None
  fake_contract.validate_values=lambda *_:self.fail('Original validator called')
  bound=types.SimpleNamespace(arguments={k:object()for k in fake_contract.NAMES})
  with patch.dict(sys.modules,{'torch':fake_torch,'tileir_vllm_post_update_v3.contract':fake_contract}):
   with self.assertRaisesRegex(ValueError,'Sentinel'):owner.call('post_update',None,bound)
  self.assertEqual(len(seen),1);self.assertEqual(len(owner.rows),1)
  binding.close();self.assertNotIn('call',vars(owner));self.assertIs(owner.call.__func__,ns['call'])
 def test_close_on_tampered_method(self):
  from .binding import Binding
  owner=type('Owner',(),{})();owner.call=object();b=Binding(owner,None);b.copied=object();b.installed=True
  with self.assertRaisesRegex(RuntimeError,'concurrently'):b.close()
  self.assertNotIn('call',vars(owner));self.assertFalse(b.installed)

if __name__=='__main__':unittest.main()
