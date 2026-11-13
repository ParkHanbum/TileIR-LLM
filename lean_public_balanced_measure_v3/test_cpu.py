import ast,copy,json,os,subprocess,sys,types,unittest,tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from lean_public_balanced_measure_v3 import design,gates,request,scopes
from lean_public_balanced_measure_v3.profile_contract import inspect_trace,scope_names
from lean_public_balanced_measure_v3.test_contract_cpu import example_trace
from lean_public_composition_v2.adapter import ORDER
from lean_public_runtime_qualification_v2.test_cpu import fixture
R=Path(__file__).resolve().parent;P=R.parent
class DesignTests(unittest.TestCase):
 def test_explicit_two_each(self):
  for i in(0,1):self.assertEqual(design.orders(i).count(['native','lean']),2);self.assertEqual(design.orders(i).count(['lean','native']),2)
 def test_complementary(self):self.assertEqual(design.orders(1),[list(reversed(x))for x in design.orders(0)]);self.assertEqual(design.profile_order(1),list(reversed(design.profile_order(0))))
 def test_wrong_sequence_rejected(self):
  pairs=[dict(order=['native','lean'],requests=[dict(mode='native'),dict(mode='lean')])for _ in range(4)]
  with self.assertRaises(AssertionError):design.validate_schedule(0,pairs)
 def test_current_proof_state_explicit(self):
  if json.loads((R/'QUALIFICATION-RECEIPT.json').read_text()).get('complete') is True:self.assertIn('evidence',gates.qualification(R))
  else:
   with self.assertRaisesRegex(AssertionError,'Lean25 actual correctness pending'):gates.qualification(R)
class RequestTests(unittest.TestCase):
 def exercise(self,mode,measured=True,fail=False):
  ft,fp,_=gates.frozen_reference();events=[];lifecycle={'installed':False};model=types.SimpleNamespace();params=object();keys=[]
  class Adapter:
   ATTR='lean';
   def install(self,m):events.append('install');setattr(m,self.ATTR,True);return dict(install_order=list(ORDER))
   def start(self,m,mode):events.append('start');selfmode=mode;assert selfmode=='lean'
   def finish(self,m):events.append('finish');return fixture()
   def close(self,m):events.append('close');return dict(restored=True,close_order=list(reversed(ORDER)),components={k:dict(restored=True)for k in ORDER})
   def release(self,m):events.append('release');delattr(m,self.ATTR)
  adapter=Adapter()
  class Engine:
   unfinished=False
   def has_unfinished_requests(self):return self.unfinished
   def apply_model(self,fn):return[fn(model)]
   def add_request(self,k,inp,par):
    assert par is params and inp==dict(prompt_token_ids=list(range(53)));events.append('add');keys.append(k);self.unfinished=True
   def step(self):
    events.append('step')
    if fail:raise RuntimeError('CUDA out of memory mock')
    self.unfinished=False
    return[types.SimpleNamespace(finished=True,request_id=k,outputs=[types.SimpleNamespace(token_ids=ft[i],logprobs=[{int(k):types.SimpleNamespace(**v)for k,v in row.items()}for row in fp[i]])])for i,k in enumerate(keys)]
  cuda=types.SimpleNamespace(synchronize=lambda:events.append('sync'),reset_peak_memory_stats=lambda:events.append('resetpeak'),max_memory_allocated=lambda:100,max_memory_reserved=lambda:200)
  torch=types.SimpleNamespace(cuda=cuda,backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_bf16_reduced_precision_reduction_split_k=True))))
  policy=types.SimpleNamespace(select=lambda arm:('native',True));clock_count=[0]
  def clock():clock_count[0]+=1;events.append('clock'+str(clock_count[0]));return clock_count[0]*1000000
  def record(x):events.append('record')
  if fail:
   with self.assertRaisesRegex(RuntimeError,'out of memory'):request.request(Engine(),torch,policy,adapter,lifecycle,list(range(53)),params,'test',mode,measured,ft,fp,record,clock)
   self.assertTrue(lifecycle['installed']);self.assertNotIn('release',events);return
  out=request.request(Engine(),torch,policy,adapter,lifecycle,list(range(53)),params,'test',mode,measured,ft,fp,record,clock)
  self.assertFalse(lifecycle['installed']);self.assertEqual(out['whole_ms'],1 if measured else None)
  if measured:
   self.assertLess(events.index('resetpeak'),events.index('clock1'));self.assertLess(events.index('clock1'),events.index('add'));self.assertLess(events.index('step'),events.index('clock2'));self.assertLess(events.index('clock2'),events.index('record'))
  else:self.assertEqual(clock_count[0],0)
  if mode=='lean':
   self.assertLess(events.index('install'),events.index('start'));self.assertLess(events.index('record'),events.index('finish'));self.assertLess(events.index('finish'),events.index('close'));self.assertEqual(events[events.index('close')+1],'sync');self.assertLess(events.index('close'),events.index('release'))
  else:self.assertNotIn('install',events);self.assertIsNone(out['selected_dispatch'])
  return out
 def test_native_endpoint(self):self.exercise('native')
 def test_lean_endpoint_restore(self):self.exercise('lean')
 def test_untimed_no_clock(self):self.exercise('lean',False)
 def test_oom_no_retry_preserves_cleanup_state(self):self.exercise('lean',fail=True)
class ScopeTests(unittest.TestCase):
 def test_actual_base_call_scope_and_restore(self):
  from lean_public_composition_v1.support import Base
  old=Base.call;events=[]
  @contextmanager
  def record(name):events.append(name);yield
  torch=types.SimpleNamespace(profiler=types.SimpleNamespace(record_function=record));controller=types.SimpleNamespace(active=True,failed=None,latch=lambda k,e:None);b=Base(controller,'norm',2)
  with scopes.public_api_scopes(torch,True):
   self.assertIsNot(Base.call,old);x=object();self.assertIs(b.call('owner','prefill',lambda y:y,x),x)
  self.assertIs(Base.call,old);self.assertEqual(events,['tileir_public_api::norm'])
 def test_exception_restores(self):
  from lean_public_composition_v1.support import Base
  old=Base.call
  with self.assertRaises(ValueError):
   with scopes.public_api_scopes(None,True):raise ValueError('mock')
  self.assertIs(Base.call,old)
 def test_disabled_wall_no_patch(self):
  from lean_public_composition_v1.support import Base
  old=Base.call
  with scopes.public_api_scopes(None,False):self.assertIs(Base.call,old)
class AuditTests(unittest.TestCase):
 def result(self,index=0):
  ft,fp,options=gates.frozen_reference()
  def rec(mode,measured=False,profile=False):
   x=dict(mode=mode,measured=measured,whole_ms=1.0 if measured else None,output_token_ids=ft,public_logprobs=fp,allow_bf16_reduced_precision_reduction=True,allow_bf16_reduced_precision_reduction_split_k=True,selected_dispatch=None)
   if mode=='lean':x.update(selected_dispatch=fixture(),adapter_install=[dict(install_order=list(ORDER))],adapter_restore=[dict(restored=True,close_order=list(reversed(ORDER)),components={k:dict(restored=True)for k in ORDER})],adapter_release_absent=[True])
   if profile:
    tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);trace=Path(tmp.name)/'trace.json';trace.write_text(json.dumps(example_trace(mode)))
    x.update(profiled=True,trace_path=str(trace),trace_sha256=gates.sha(trace),trace_bytes=trace.stat().st_size,profile_scope_names=scope_names(mode),profile_partition=inspect_trace(trace,mode))
   return x
  return dict(complete=True,phase='complete',candidate_checks_passed=True,engine_shutdown=True,process_index=index,baseline='current eager vLLM mode0/graphsNONE',engine_options=options,cache_preseed=dict(destination=str(R/('process'+str(index))/'cache_v1'),files_verified=True,separate_inodes_verified=True),precision_policy=dict(initial=dict(bf16=True,split_k=True),restored=True,restored_split_k=True),qualification_requests=[rec('native'),rec('lean')],pairs=[dict(order=o,requests=[rec(m,True)for m in o])for o in design.orders(index)],profile_requests=[rec(m,profile=True)for m in design.profile_order(index)],final_adapter_absent=[True],profile_scope_restored=True)
 def test_valid_both(self):
  for i in(0,1):self.assertTrue(gates.audit(self.result(i),R/('process'+str(i)))['passed'])
 def reject(self,key,change):
  r=self.result();change(r);self.assertFalse(gates.audit(r,R/'process0')['passed'],key)
 def test_unbalanced_reject(self):self.reject('order',lambda r:r['pairs'][1].update(order=['native','lean']))
 def test_profile_timing_reject(self):self.reject('profiletime',lambda r:r['profile_requests'][0].update(measured=True,whole_ms=5))
 def test_lean_scope_not_restored(self):self.reject('scope',lambda r:r.update(profile_scope_restored=False))
 def test_trace_changed(self):self.reject('trace',lambda r:r['profile_requests'][0].update(trace_sha256='bad'))
 def test_cleanup_error(self):self.reject('cleanup',lambda r:r.update(cleanup_errors=['error']))
 def test_nan_wall(self):self.reject('nan',lambda r:r['pairs'][0]['requests'][0].update(whole_ms=float('nan')))
 def test_wrong_cache_process(self):self.reject('cache',lambda r:r['cache_preseed'].update(destination=str(R/'process1/cache_v1')))
class SourceTests(unittest.TestCase):
 def test_dry_import_no_cuda(self):
  for name in('run.py','launch.py'):
   command="import sys,runpy;sys.argv=["+repr(name)+",'--process-index','1'];runpy.run_path("+repr(str(R/name))+",run_name='__main__');assert 'torch'not in sys.modules and 'cuda.tile'not in sys.modules and 'vllm'not in sys.modules"
   p=subprocess.run([sys.executable,'-B','-c',command],cwd=P,capture_output=True,timeout=30,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(P)})
   self.assertEqual(p.returncode,0,p.stderr.decode());self.assertIsInstance(json.loads(p.stdout),dict)
 def test_ast_all(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
 def test_no_gpu_artifacts(self):
  self.assertFalse((R/'process0').exists());self.assertFalse((R/'process1').exists())
 def test_profiler_minimal(self):
  s=(R/'run.py').read_text();self.assertIn('record_shapes=False,profile_memory=False,with_stack=False,with_flops=False',s);self.assertNotIn('TorchDispatchMode',s);self.assertIn('assert Base.call is base_call',s)
 def test_parent_cache_distinct(self):
  s=(R/'launch.py').read_text();self.assertIn("R=S/('process'+str(args.process_index))",s);self.assertIn('cache_preseed=preflight(R)',s);self.assertIn("audit(result,R)",s)
if __name__=='__main__':unittest.main()
