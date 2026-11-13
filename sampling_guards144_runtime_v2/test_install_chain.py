"""Execute unmodified production installers, real original function/code objects.
Only import containers and the vLLM Sampler class are CPU shells; no numerical
function, source provenance, factory, installer, bind or restore is mocked.
"""
import ast,hashlib,inspect,json,sys,textwrap,types,unittest
from pathlib import Path
from unittest.mock import patch
from . import range_owned,owned
P=Path(__file__).resolve().parent.parent

def load(path,names,env):
 tree=ast.parse(path.read_text());nodes=[x for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
 assert len(nodes)==len(names),(path,names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',dont_inherit=True),env)
 return [env[n]for n in names]
def module(name,**attrs):return types.ModuleType(name).__dict__ | attrs
class Chain(unittest.TestCase):
 def setUp(self):
  env=dict(Counter=__import__('collections').Counter,functools=__import__('functools'),inspect=inspect,json=json,Path=Path,hashlib=hashlib,textwrap=textwrap)
  Base,phase=load(P/'lean_public_composition_v1/support.py',['Base','phase_index'],env)
  env['Base']=Base
  S,Three=load(P/'lean_public_composition_v1/sampler.py',['_Sampler','SamplerThree'],env)
  env.update(_Sampler=S,SamplerThree=Three)
  Four=load(P/'lean_public_composition_v2/sampler.py',['SamplerFour'],env)[0]
  e=dict(env,Original=Four,R=P/'lean_finite_guard_runtime_v1/sampler.py')
  e['R']=e['R'].parent
  Top=load(e['R']/'sampler.py',['SamplerFour'],e)[0]
  from finite_six_site_cpu_v1.factory import build
  e=dict(env,Top5Qualified=Top,build=build)
  Finite,Orig=load(P/'lean_finite_guard_runtime_v4/bindings.py',['FiniteSites','SamplerFour'],e)
  e=dict(OriginalSampler=Orig,prepare=range_owned.prepare)
  Range=load(Path(range_owned.__file__).with_name('range_adapter.py'),['Sampler'],e)[0]
  e=dict(ranged=types.SimpleNamespace(Sampler=Range),prepare=owned.prepare)
  self.Sample=load(Path(owned.__file__).with_name('adapter.py'),['Sampler'],e)[0]
  self.Range=Range;self.Orig=Orig
  mods={}
  def add(name,**attrs):
   mod=types.ModuleType(name);mod.__dict__.update(attrs);mods[name]=mod;return mod
  def api(folder,filename,name):
   d={};fn=load(P/folder/filename,[name],d)[0];add(folder,**{name:fn});return fn
  ranks=api('tileir_vllm_selected_ranks_v1','api.py','selected_token_ranks')
  scores=api('tileir_vllm_selected_logprobs_v2','api.py','compute_token_logprobs')
  top=api('tileir_vllm_top5_v2','api.py','top5_indices')
  cat=api('tileir_vllm_sampler_cat_v1','api.py','concat_sampled_top5')
  require=api('tileir_validation_primitives_v2','api.py','require_finite');add('tileir_validation_primitives_v2.api',require_finite=require)
  root=P/'sampler_four_runtime_integration_v1';spec=json.loads((root/'SOURCE.json').read_text())
  helper=load(Path(spec['path']),['compute_topk_scores'],{'torch':types.SimpleNamespace(Tensor=object),'LogprobsTensors':object})[0]
  def identity(fn):return {'source_sha256':hashlib.sha256(Path(fn.__code__.co_filename).read_bytes()).hexdigest()}
  # install's original env has independent copy; update that exact mapping.
  Four.install.__globals__['identity']=identity
  builder=load(root/'adapter.py',['build_copy'],dict(ROOT=root,Path=Path,json=json,hashlib=hashlib))[0]
  add('sampler_four_runtime_integration_v1.adapter',build_copy=builder,ROOT=root)
  class FakeSampler:
   def __call__(self,*a,**k):raise AssertionError('No model invocation in CPU install test')
  self.owner=FakeSampler;self.original_call=FakeSampler.__call__
  sam=add('vllm.v1.worker.gpu.sample.sampler',Sampler=FakeSampler,compute_topk_scores=helper)
  log=add('vllm.v1.worker.gpu.sample.logprob',compute_topk_scores=helper)
  add('vllm.v1.worker.gpu.sample',sampler=sam,logprob=log)
  self.sam=sam;self.helper=helper;self.mods=mods
  cb=lambda *a:True
  self.controller=types.SimpleNamespace(active=False,failed=None,range48=types.SimpleNamespace(predicate=cb),sampling96=types.SimpleNamespace(predicate=cb,gather=cb))
 def test_real_install_and_reverse_close(self):
  with patch.dict(sys.modules,self.mods):
   s=self.Sample(self.controller,'sampler_four',16);s.install(None)
   body=s.candidate.__globals__['_tileir_selected_ranks'].__kwdefaults__['_body']
   self.assertEqual(body.__code__.co_filename,str(Path(owned.__file__).with_name('ranks_api.candidate.py.txt')))
   self.assertTrue(callable(body.__globals__['_finite_all']));self.assertIs(body.__globals__['_range_guard'],self.controller.range48.predicate)
   s.close();self.assertIs(self.owner.__call__,self.original_call);self.assertIs(self.sam.compute_topk_scores,self.helper)
 def test_partial_install_failure_restore(self):
  with patch.dict(sys.modules,self.mods):
   s=self.Sample(self.controller,'sampler_four',16)
   # Fail second range copy only, after real production base install + first copy.
   real=self.Range.install.__globals__['prepare']
   def fail(fn,site,cb):
    if site=='scores':raise AssertionError('Injected post-ranks failure')
    return real(fn,site,cb)
   with patch.dict(self.Range.install.__globals__,prepare=fail):
    with self.assertRaises(AssertionError):s.install(None)
   self.assertEqual(len(s.range_bindings),1)
   s.close();self.assertIs(self.owner.__call__,self.original_call);self.assertIs(self.sam.compute_topk_scores,self.helper)
 def test_old_installer_fails_real_chain_and_restores(self):
  from i64_range48_runtime_v1.owned import prepare as old_prepare
  with patch.dict(sys.modules,self.mods):
   s=self.Sample(self.controller,'sampler_four',16)
   with patch.dict(self.Range.install.__globals__,prepare=old_prepare):
    with self.assertRaises(AssertionError):s.install(None)
   self.assertEqual(s.range_bindings,[])
   s.close();self.assertIs(self.owner.__call__,self.original_call);self.assertIs(self.sam.compute_topk_scores,self.helper)
 def test_finite_code_tamper_rejected(self):
  with patch.dict(sys.modules,self.mods):
   s=self.Orig(self.controller,'sampler_four',16);s.install(None)
   fn=s.candidate.__globals__['_tileir_selected_ranks'].__kwdefaults__['_body']
   self.assertEqual(fn.__code__.co_firstlineno,2)
   range_owned.prepare(fn,'ranks',self.controller.range48.predicate)
   fn.__code__=fn.__code__.replace(co_firstlineno=1)
   with self.assertRaises(AssertionError):range_owned.prepare(fn,'ranks',None)
   s.close()
if __name__=='__main__':unittest.main()
