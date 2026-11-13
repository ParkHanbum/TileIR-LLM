import ast,pathlib,types,sys,unittest
from unittest.mock import patch
from .isolated import cases
P=pathlib.Path(__file__).resolve().parent.parent;R=pathlib.Path(__file__).resolve().parent
class Tests(unittest.TestCase):
 def test_cases(self):
  c=cases();self.assertEqual((len(c)+1)*2,52);self.assertEqual(len({x['name']for x in c}),len(c));self.assertEqual(sum(x.get('reject',False)for x in c),4)
 def test_fp32_before_gather(self):
  from public_temperature_zero_v1 import api
  class T: dtype='bf16'
  fake=types.SimpleNamespace(Tensor=T,float32='fp32')
  with patch.dict(sys.modules,torch=fake),patch.object(api,'selected_1d_checked',side_effect=AssertionError('gather called')):
   with self.assertRaises(ValueError):api.selected_all_zero(T(),None)
 def test_metadata(self):
  from public_temperature_zero_v1.contract import vector
  self.assertEqual(vector((32,),(1,),'torch.float32',0,128),(32,4))
  with self.assertRaises(ValueError):vector((32,),(2,),'torch.float32',0,256)
  with self.assertRaises(ValueError):vector((32,),(1,),'torch.float32',1,128)
 def test_lifecycle(self):
  t=ast.parse((R/'adapter.py').read_text());n=next(x for x in t.body if isinstance(x,ast.ClassDef)and x.name=='TempChecks');env={}
  exec(compile(ast.Module(body=[n],type_ignores=[]),'checks','exec'),env);c=types.SimpleNamespace(failed=None);x=env['TempChecks'](c)
  with self.assertRaises(AssertionError):x.start('replace')
  x.start('shadow')
  with self.assertRaises(AssertionError):x.finish()
  x.rows=[{'stage':'verified'}for _ in range(16)];x.finish();x.start('replace');self.assertEqual(x.rows,[])
 def test_conditional_engine_order(self):
  s=(R/'run.py').read_text();self.assertLess(s.index("result['isolated_temperature']=isolated_run()"),s.index('engine=LLMEngine.from_engine_args'));self.assertIn("assert result['isolated_temperature']['complete']",s)
 def test_owned_body_order(self):
  s=(P/'sampling_guards144_runtime_v2/range_greedy.candidate.py.txt').read_text();self.assertEqual(s.count('bool((temp.index_select(0,idx)==0).all())'),1)
  self.assertLess(s.index("_range_guard('greedy'"),s.index('temp.index_select'))
 def test_phase_labels(self):
  t=ast.parse((R/'adapter.py').read_text());cl=next(n for n in t.body if isinstance(n,ast.ClassDef)and n.name=='Controller')
  class Base:
   def finish(self):return {'sampling96':{'temperature_original_retained':16}}
  env={'lean':types.SimpleNamespace(Controller=Base)}
  exec(compile(ast.Module(body=[cl],type_ignores=[]),'controller','exec'),env)
  for mode,want in [('shadow',16),('replace',0)]:
   obj=env['Controller'].__new__(env['Controller']);obj.temperature16=types.SimpleNamespace(finish=lambda:{'mode':mode})
   d=obj.finish();self.assertEqual(d['sampling96']['temperature_original_retained'],want);self.assertIs(d['sampling96']['temperature_followup_hook'],True)
 def test_parse(self):
  for folder in (R,P/'public_temperature_zero_v1'):
   for p in folder.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()

class GreedyChain(unittest.TestCase):
 def test_actual_install_restore_and_callbacks(self):
  import inspect,json,hashlib,functools
  from collections import Counter
  from sampling_guards144_runtime_v2.test_install_chain import load
  from sampling_guards144_runtime_v2 import range_owned,owned
  env=dict(Counter=Counter,functools=functools,inspect=inspect,json=json,Path=pathlib.Path)
  Base,phase=load(P/'lean_public_composition_v1/support.py',['Base','phase_index'],env)
  env['Base']=Base
  _,Orig=load(P/'lean_public_composition_v1/sampler.py',['_Sampler','Greedy'],env)
  env2=dict(OriginalGreedy=Orig,prepare=range_owned.prepare)
  Range=load(P/'sampling_guards144_runtime_v2/range_adapter.py',['Greedy'],env2)[0]
  Sample=load(P/'sampling_guards144_runtime_v2/adapter.py',['Greedy'],dict(ranged=types.SimpleNamespace(Greedy=Range),prepare=owned.prepare))[0]
  e=dict(proof=types.SimpleNamespace(Greedy=Sample,__file__=str(P/'sampling_guards144_runtime_v2/adapter.py')),Path=pathlib.Path,hashlib=hashlib,inspect=inspect,textwrap=__import__('textwrap'),R=R)
  Temp=load(R/'adapter.py',['Greedy'],e)[0]
  gp=pathlib.Path('/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/vllm/v1/worker/gpu/sample/gumbel.py')
  original=load(gp,['gumbel_sample'],{'torch':types.SimpleNamespace(Tensor=object)})[0]
  api=load(P/'tileir_vllm_greedy_sample_v1/api.py',['greedy_sample'],{})[0]
  Orig.install.__globals__['identity']=lambda f:{'source_sha256':hashlib.sha256(pathlib.Path(f.__code__.co_filename).read_bytes()).hexdigest()}
  class S:
   def __call__(self,*a,**k):raise AssertionError('No execution')
  old=S.__call__;sam=types.ModuleType('sampler');sam.Sampler=S;sam.gumbel_sample=original
  pkg=types.ModuleType('vllm.v1.worker.gpu.sample');pkg.sampler=sam
  ap=types.ModuleType('tileir_vllm_greedy_sample_v1');ap.greedy_sample=api
  cb=lambda *a:True;c=types.SimpleNamespace(failed=None,active=False,range48=types.SimpleNamespace(predicate=cb),sampling96=types.SimpleNamespace(predicate=cb,gather=cb),temperature16=types.SimpleNamespace(predicate=cb))
  with patch.dict(sys.modules,{'vllm.v1.worker.gpu.sample':pkg,'tileir_vllm_greedy_sample_v1':ap}):
   obj=Temp(c,'greedy_sample',16);obj.install(None)
   self.assertIs(obj.range_body.__globals__['_range_guard'],cb);self.assertIs(obj.range_body.__globals__['_temperature_zero'],cb);self.assertIs(obj.candidate.__globals__['_sampling_predicate'],cb)
   self.assertEqual(obj.range_body.__code__.co_filename,str(R/'temperature_owned.py.txt'))
   obj.close();self.assertIs(S.__call__,old);self.assertIs(sam.gumbel_sample,original)
