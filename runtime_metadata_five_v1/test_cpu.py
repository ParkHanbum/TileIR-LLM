import ast,copy,json,sys,unittest,__future__
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from runtime_metadata_five_v1 import api,binding
from runtime_metadata_five_v1.qualifier import Qualifier,LIMITS
class Tests(unittest.TestCase):
 def test_exact_actual_sources(self):
  for key,e in binding.EXPECTED.items():
   tree=ast.parse(Path(e['path']).read_text());nodes=tree.body if e['owner'] is None else next(n.body for n in tree.body if isinstance(n,ast.ClassDef) and n.name==e['owner'])
   node=copy.deepcopy(next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==e['name']))
   module=ast.Module(body=[node],type_ignores=[]);ns={};exec(compile(module,e['path'],'exec',flags=__future__.annotations.compiler_flag),ns)
   self.assertEqual(ast.dump(ast.parse(binding.checked(ns[e['name']],key)),include_attributes=False),e['ast'])
   binding.copy(ns[e['name']],key,{})
 def test_transforms_exact_counts(self):
  for key,e in binding.EXPECTED.items():
   output=binding.transform(e['source'],key)
   funcs=[n.func.id for n in ast.walk(output) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id.startswith('_meta_')]
   self.assertEqual(len(funcs),2 if key!='gdn' else 1)
 def test_missing_target_rejected(self):
  with self.assertRaises(ValueError):binding.transform('def a(): return 1','common')
 def test_cache_semantics(self):
  e=binding.EXPECTED['common'];t=binding.transform(e['source'],'common');ns={'_meta_subq':lambda a,b:[x-y for x,y in zip(a,b)],'_meta_subs':lambda a,b:[x-y for x,y in zip(a,b)]};exec(compile(t,'<cpu>','exec',flags=__future__.annotations.compiler_flag),ns)
  owner=SimpleNamespace(query_start_loc=list(range(33)),seq_lens=[53]*32,_num_computed_tokens_cache=None)
  x=ns[e['name']](owner);self.assertEqual(x,[52]*32);self.assertIs(ns[e['name']](owner),x)
 def test_wrap_reference(self):
  wrap=lambda v:(v+2**31)%2**32-2**31
  self.assertEqual(wrap(-2147483648-1),2147483647);self.assertEqual(wrap(2147483647-(-1)),-2147483648)
 def test_full_scope_reject_before_framework(self):
  with self.assertRaises(ValueError):api.full((2047,),-1,dtype='torch.int32',device=SimpleNamespace(type='cuda'))
  with self.assertRaises(ValueError):api.full((2048,),0,dtype='torch.int32',device=SimpleNamespace(type='cuda'))
 def test_latch_entry(self):
  q=Qualifier(None);events=[]
  def bad():raise ValueError('failure')
  with self.assertRaises(ValueError):q.guard(bad)()
  with self.assertRaises(RuntimeError):q.guard(lambda:events.append(1))()
  self.assertEqual(events,[])
 def test_phase_gate(self):
  q=Qualifier(None)
  with self.assertRaises(RuntimeError):q.start('replace')
  q=Qualifier(None);q.start('shadow');q.rows=[{'label':k}for k,v in LIMITS.items()for _ in range(v)]
  self.assertEqual(q.finish()['calls'],5);q.start('replace')
 def test_precall_cap(self):
  q=Qualifier(None);q.start('shadow');q.rows=[dict(label='subq')]
  with self.assertRaises(RuntimeError):q.invoke('subq',(),lambda:None,lambda:None,None)
  self.assertTrue(q.failed)
 def test_restore_best_effort(self):
  a=SimpleNamespace(f=lambda:1);b=SimpleNamespace(g=lambda:2);old=a.f;new=lambda:3;foreign=lambda:4;a.f=new
  x=object.__new__(binding.Bindings);x.installed=[(a,'f',old,new),(b,'g',lambda:2,foreign)]
  with self.assertRaises(RuntimeError):x.close()
  self.assertIs(a.f,old)
if __name__=='__main__':unittest.main()
