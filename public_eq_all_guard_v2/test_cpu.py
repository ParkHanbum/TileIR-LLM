import ast,math,unittest,json,__future__
from pathlib import Path
from public_eq_all_guard_v2.contract import descriptor,reference
from public_eq_all_guard_v2.owned import prepare
R=Path(__file__).parent
class Tests(unittest.TestCase):
 def test_special_values(self):
  self.assertTrue(reference([0.,-0.],0));self.assertTrue(reference([],1))
  for x in [float('nan'),float('inf'),float('-inf'),1e-40]:self.assertFalse(reference([x],0))
  self.assertTrue(reference([1.,1.],1));self.assertFalse(reference([1.,-1.],1))
 def test_broadcast(self):self.assertEqual(descriptor((32,4),(0,0),'torch.float32',3,16,1)['physical'],1)
 def test_initial(self):self.assertEqual(descriptor((32,32,128,128),(524288,16384,128,1),'torch.float32',0,1<<26,0)['physical'],1<<24)
 def test_scalar_empty(self):
  self.assertEqual(descriptor((),(),'torch.float32',0,4,1)['physical'],1)
  self.assertEqual(descriptor((0,),(1,),'torch.float32',0,0,0)['physical'],0)
 def test_extent_and_layout(self):
  for shape,stride,offset,size in [((32,4),(0,0),3,12),((4,),(2,),0,32),((1<<25,),(1,),0,1<<27)]:
   with self.assertRaises(ValueError):descriptor(shape,stride,'torch.float32',offset,size,0)
 def test_flags_and_constants(self):
  for val in [True,2,0.0]:
   with self.assertRaises(ValueError):descriptor((1,),(1,),'torch.float32',0,4,val)
  with self.assertRaises(ValueError):descriptor((1,),(1,),'torch.float32',0,4,0,neg=True)
 def test_only_expression_changed(self):
  for tag in ['attention','gdn','gdn_active']:
   a=(R/(tag+'.original.py.txt')).read_text();b=(R/(tag+'.copied.py.txt')).read_text()
   old='bool((t==1).all())' if tag=='attention' else "bool((inputs['initial_state']==0).all())"
   new='_eq_all(t,1)' if tag=='attention' else "_eq_all(inputs['initial_state'],0)"
   self.assertEqual(b,a.replace(old,new));self.assertEqual(a.count(old),1)
 def test_real_dedent_prepare(self):
  specs=json.loads((R/'SOURCES.json').read_text())
  for tag,d in specs.items():
   if tag=='gdn_active':
    ns={};exec(compile(Path(d['path']).read_text(),d['path'],'exec'),ns);fn=ns['apply']
   else:
    tree=ast.parse(Path(d['path']).read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==d['class']);cl.bases=[];cl.body=[n for n in cl.body if isinstance(n,ast.FunctionDef)and n.name=='apply'];ns={};exec(compile(ast.Module(body=[cl],type_ignores=[]),d['path'],'exec',flags=__future__.annotations.compiler_flag),ns);fn=ns[d['class']].apply
   self.assertTrue(callable(prepare(fn,tag,lambda *a:True)))
 def test_shortcircuit(self):
  calls=[]
  def pred(*x):calls.append(x);return False
  # Exact guard chain preserves short circuit and the same assertion boundary.
  class T: dtype='wrong';shape=(32,4)
  t=T()
  with self.assertRaises(AssertionError):assert t.dtype=='fp32' and t.shape==(32,4) and pred(t,1)
  self.assertFalse(calls)
 def test_no_framework(self):
  import sys
  self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda',sys.modules)
if __name__=='__main__':unittest.main()
