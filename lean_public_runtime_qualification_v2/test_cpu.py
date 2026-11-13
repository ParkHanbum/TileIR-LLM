import ast,copy,json,os,subprocess,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
from lean_public_runtime_qualification_v2 import gates,cleanup
from lean_public_composition_v2.adapter import SPECS,ORDER
R=Path(__file__).resolve().parent;P=R.parent
class OutputTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.tokens,cls.public,cls.options=gates.frozen_reference()
 def test_exact_frozen(self):
  self.assertEqual(gates.exact_tokens(self.tokens,self.tokens)['values'],512);self.assertEqual(gates.exact_public(self.public,self.public)['values'],2560)
 def test_token_failure(self):
  b=copy.deepcopy(self.tokens);b[31][15]+=1
  with self.assertRaises(AssertionError):gates.exact_tokens(self.tokens,b)
 def test_public_numeric_no_tolerance(self):
  b=copy.deepcopy(self.public);key=next(iter(b[0][0]));b[0][0][key]['logprob']+=1e-12
  with self.assertRaises(AssertionError):gates.exact_public(self.public,b)
 def test_rank_failure(self):
  b=copy.deepcopy(self.public);key=next(iter(b[0][0]));b[0][0][key]['rank']+=1
  with self.assertRaises(AssertionError):gates.exact_public(self.public,b)
 def test_nonfinite_failure(self):
  b=copy.deepcopy(self.public);b[0][0][next(iter(b[0][0]))]['logprob']=float('nan')
  with self.assertRaises(AssertionError):gates.exact_public(self.public,b)
 def test_exact_key_count(self):
  b=copy.deepcopy(self.public);b[0][0]['invalid']=dict(logprob=0,rank=6)
  with self.assertRaises(AssertionError):gates.exact_public(self.public,b)
 def test_actual_qualification(self):self.assertIn('evidence',gates.qualification(R))
def fixture():
 parts={k:dict(calls=n,selected_native_calls=0,selected_fallback_calls=0,owners=[dict(owner=k,phase='prefill',calls=n)])for k,c,n in SPECS}
 g=parts['all_gemm'];g.update(k128_calls=360,existing_calls=2088,owners=[dict(owner=x['path'],phase=p,calls=n)for x in gates.plan()['selected']for p,n in [('prefill',1),('decode',15)]])
 parts['indexed_decode_conv']['delegated_prefill_tileir']=24
 parts['sampler_four'].update(logical_suboperation_calls=64,suboperation_counts=dict(top5=16,ranks=16,selected_logprobs=16,cat=16),native_cat_retained=False)
 parts['runner_padding']['logical_suboperation_calls']=64
 parts['gdn_parent_state']['logical_suboperation_calls']=96
 return dict(components=parts,selected_boundaries=5888,selected_logical_operations=6056,selected_native_calls=0,selected_fallback_calls=0,failed=None)
class DispatchTests(unittest.TestCase):
 def test_all13_2448(self):
  d=gates.dispatch(fixture());self.assertEqual(len(d['gemm_contract_calls']),13);self.assertEqual(sum(d['gemm_contract_calls'].values()),2448)
 def test_missing_part(self):
  d=fixture();d['components'].pop('norm')
  with self.assertRaises(AssertionError):gates.dispatch(d)
 def test_count_wrong(self):
  d=fixture();d['components']['norm']['calls']-=1
  with self.assertRaises(AssertionError):gates.dispatch(d)
 def test_fallback_wrong(self):
  d=fixture();d['components']['all_gemm']['selected_fallback_calls']=1
  with self.assertRaises(AssertionError):gates.dispatch(d)
 def test_gemm_owner_wrong(self):
  d=fixture();d['components']['all_gemm']['owners'][0]['owner']='wrong'
  with self.assertRaises(AssertionError):gates.dispatch(d)
 def test_k128_wrong(self):
  d=fixture();d['components']['all_gemm']['k128_calls']=359
  with self.assertRaises(AssertionError):gates.dispatch(d)
 def test_prefill_delegation_wrong(self):
  d=fixture();d['components']['indexed_decode_conv']['delegated_prefill_tileir']=0
  with self.assertRaises(AssertionError):gates.dispatch(d)
class ResultGateTests(unittest.TestCase):
 def result(self):
  tokens,public,options=gates.frozen_reference()
  def arm(mode):return dict(mode=mode,output_token_ids=tokens,public_logprobs=public,selected_dispatch=fixture()if mode=='lean'else None,allow_bf16_reduced_precision_reduction=True,allow_bf16_reduced_precision_reduction_split_k=True)
  return dict(complete=True,candidate_checks_passed=True,phase='complete',gpu_executed=True,scope='current-eager native-to-lean correctness only',timed_requests=0,engine_options=options,cache_preseed=dict(destination=str(R/'cache_v1'),files_verified=True,separate_inodes_verified=True),precision_policy=dict(initial=dict(bf16=True,split_k=True),restored=True,restored_split_k=True),adapter_install=[dict(install_order=list(ORDER))],native_request=arm('native'),lean_request=arm('lean'),adapter_restore=[dict(restored=True,close_order=list(reversed(ORDER)),components={k:dict(restored=True)for k in ORDER})],synchronized_before_close=True,synchronized_after_close=True,adapter_released=True,adapter_release_absent=[True],engine_shutdown=True)
 def test_valid_fixture(self):self.assertTrue(gates.audit(self.result())['passed'])
 def test_wrong_cache_destination(self):
  r=self.result();r['cache_preseed']['destination']='/wrong/cache_v1';self.assertFalse(gates.audit(r)['passed'])
 def test_wrong_reverse_restore(self):
  r=self.result();r['adapter_restore'][0]['close_order']=list(ORDER);self.assertFalse(gates.audit(r)['passed'])
 def test_owner_release_missing(self):
  r=self.result();r['adapter_release_absent']=[False];self.assertFalse(gates.audit(r)['passed'])
 def test_historical_optimized_baseline_rejected(self):
  r=self.result();r['engine_options']['compilation_config']={'mode':3,'cudagraph_mode':'FULL'};self.assertFalse(gates.audit(r)['passed'])
 def test_policy_change_rejected(self):
  r=self.result();r['precision_policy']['initial']['bf16']=False;self.assertFalse(gates.audit(r)['passed'])
 def test_cleanup_error_invalidates_complete(self):
  r=self.result();r['cleanup_errors']=[{'stage':'close','error':'test'}];self.assertFalse(gates.audit(r)['passed'])

class CleanupTests(unittest.TestCase):
 def exercise(self,fail=None,bodyfail=False,installed=True):
  events=[];result={'candidate_checks_passed':not bodyfail};counter={'sync':0};adapter=types.SimpleNamespace(close=object(),release=object(),ATTR='unused')
  def event(name):
   events.append(name)
   if name==fail:raise RuntimeError(name)
  def sync():counter['sync']+=1;event('sync'+str(counter['sync']))
  def apply(fn):
   if fn is adapter.close:event('close');return[{'restored':True}]
   self.assertIs(fn.func,cleanup.release_checked);event('release');return[True]
  engine=types.SimpleNamespace(apply_model=apply,engine_core=types.SimpleNamespace(shutdown=lambda:event('shutdown')))
  policy=types.SimpleNamespace(restore=lambda:event('policy'))
  tree=ast.parse((R/'run.py').read_text());main=next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='main');actual=next(x for x in main.body if isinstance(x,ast.Try))
  module=ast.Module(body=[ast.Try(body=ast.parse("raise ValueError('body')"if bodyfail else'pass').body,handlers=[],orelse=[],finalbody=actual.finalbody)],type_ignores=[]);ast.fix_missing_locations(module)
  env=dict(final_cleanup=cleanup.final_cleanup,result=result,save=lambda:event('save'),policy=policy,engine=engine,adapter=adapter,installed=installed,sys=sys,signal=types.SimpleNamespace(alarm=lambda _:event('alarm')),__import__=lambda _:types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=sync)))
  error=None
  try:exec(compile(module,'actual_run_finally','exec'),env)
  except BaseException as e:error=e
  self.assertIn('shutdown',events);self.assertIn('alarm',events);self.assertIn('save',events)
  if fail or bodyfail:self.assertFalse(result['complete'])
  else:self.assertTrue(result['complete'])
  if bodyfail:self.assertIsInstance(error,ValueError);self.assertEqual(str(error),'body')
  elif fail:self.assertIsInstance(error,RuntimeError)
  else:self.assertIsNone(error)
  return events,result
 def test_success_order(self):
  e,r=self.exercise();self.assertEqual(e,['policy','sync1','close','sync2','release','shutdown','alarm','save']);self.assertTrue(r['adapter_released'])
 def test_close_error_no_release(self):e,r=self.exercise('close');self.assertNotIn('release',e)
 def test_sync1_error_keeps_owners(self):e,r=self.exercise('sync1');self.assertNotIn('release',e);self.assertIn('close',e)
 def test_sync2_error_keeps_owners(self):e,r=self.exercise('sync2');self.assertNotIn('release',e)
 def test_policy_error_still_cleanup(self):e,r=self.exercise('policy');self.assertIn('release',e)
 def test_release_error_still_shutdown(self):self.exercise('release')
 def test_original_exception_preserved(self):self.exercise('close',True)
 def test_preinstall_failure_no_close(self):e,r=self.exercise(bodyfail=True,installed=False);self.assertNotIn('close',e);self.assertNotIn('release',e)
class SourceTests(unittest.TestCase):
 def test_actual_dry_run_cpu(self):
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(P))
  cmd=[sys.executable,'-B','-c',"import sys,runpy;sys.argv=['run.py'];runpy.run_path("+repr(str(R/'run.py'))+",run_name='__main__');assert 'torch' not in sys.modules and 'vllm' not in sys.modules and 'cuda.tile' not in sys.modules"]
  p=subprocess.run(cmd,env=env,cwd=P,capture_output=True,timeout=30);self.assertEqual(p.returncode,0,p.stderr.decode());self.assertFalse(json.loads(p.stdout)['gpu_executed'])
 def test_actual_launcher_dry(self):
  p=subprocess.run([sys.executable,'-B',str(R/'launch.py')],cwd=P,capture_output=True,timeout=30,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'});self.assertEqual(p.returncode,0,p.stderr.decode());self.assertFalse(json.loads(p.stdout)['executed'])
 def test_native_then_install_then_lean(self):
  s=(R/'run.py').read_text();self.assertLess(s.index("native=request('native')"),s.index('engine.apply_model(adapter.install)'));self.assertLess(s.index('engine.apply_model(adapter.install)'),s.index("lean=request('lean')"))
  self.assertNotIn('adapter.start,mode=\'shadow\'',s);self.assertNotIn('perf_counter',s)
 def test_no_cache_created_and_original_api(self):
  self.assertFalse((R/'cache_v1').exists());self.assertIn('from lean_public_composition_v2 import adapter',(R/'run.py').read_text())
 def test_launch_exit_gate_and_bound(self):
  s=(R/'launch.py').read_text();self.assertIn("rec.get('returncode')==0",s);self.assertIn('child.wait(timeout=930)',s);self.assertIn('cache_preseed=preflight(R)',s)
 def test_ast(self):
  for name in('run.py','cleanup.py','gates.py','launch.py'):ast.parse((R/name).read_text())
class PrerequisiteBarrierTests(unittest.TestCase):
 def test_actual_entries_pending_before_runtime_or_admission(self):
  for entry in ('run.py','launch.py'):
   code="""import sys,runpy,importlib.abc
from pathlib import Path
original_read=Path.read_text
def pending_read(self,*a,**k):
 if self.name=='PREREQUISITE.json':return '{\"complete\":false}'
 return original_read(self,*a,**k)
Path.read_text=pending_read
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','vllm','cuda'):raise RuntimeError('FORBIDDEN_RUNTIME_IMPORT:'+fullname)
sys.meta_path.insert(0,Block())
sys.argv=[ENTRY,'--run-gpu']
try:runpy.run_path(ENTRY,run_name='__main__')
except AssertionError as e:
 assert 'Full25 actual proof pending' in str(e),repr(e)
else:raise AssertionError('pending release unexpectedly succeeded')
assert not any(x in sys.modules for x in ('torch','vllm','cuda.tile'))
""".replace('ENTRY',repr(str(R/entry)))
   p=subprocess.run([sys.executable,'-B','-c',code],cwd=P,capture_output=True,timeout=30,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
   self.assertEqual(p.returncode,0,p.stderr.decode())
 def test_worker_first_actual_gate(self):
  s=(R/'run.py').read_text();a=s.index("result['qualification_receipt']=qualification(R)")
  self.assertLess(a,s.index("held=os.fstat"));self.assertLess(a,s.index('import torch,vllm'))
  s=(R/'launch.py').read_text();self.assertLess(s.index('qualification(R)'),s.index('h.admit()'))
if __name__=='__main__':unittest.main()
