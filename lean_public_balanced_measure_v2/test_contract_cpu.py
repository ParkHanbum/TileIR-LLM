import ast,copy,json,types,unittest
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from .profile_contract import scope_names,inspect_events
from . import request
R=Path(__file__).resolve().parent

def example_trace(mode='lean'):
 names=scope_names(mode);ev=[]
 for key,ts,dur in [('install',0,8),('execution',10,10),('proof',22,8),('restore',32,4)]:ev.append(dict(ph='X',cat='user_annotation',name=names[key],pid=1,tid=2,ts=ts,dur=dur))
 for i,ts,gpu_ts,dur in [(1,12,100,2),(2,2,12,3),(3,24,15,4),(4,33,16,5)]:
  ev.append(dict(ph='X',cat='cuda_runtime',name='cudaLaunchKernel',pid=1,tid=2,ts=ts,dur=1,args={'correlation':i}))
  ev.append(dict(ph='X',cat='kernel',name='kernel'+str(i),pid=0,tid=7,ts=gpu_ts,dur=dur,args={'correlation':i}))
 return dict(traceEvents=ev)
class CorrelationTests(unittest.TestCase):
 def test_launch_not_gpu_timestamp_selects_denominator(self):
  d=inspect_events(example_trace()['traceEvents'],'lean');self.assertTrue(d['model_denominator_ready']);self.assertEqual(d['request_kernel_count'],1);self.assertEqual(d['request_kernel_duration_us'],2);self.assertEqual(d['excluded_phase_kernel_count'],3)
 def test_other_cpu_thread_does_not_enter_global_window(self):
  ev=example_trace()['traceEvents'];next(e for e in ev if e.get('cat')=='cuda_runtime'and e['args']['correlation']==1)['tid']=99
  d=inspect_events(ev,'lean');self.assertFalse(d['model_denominator_ready']);self.assertEqual(d['partition']['kernel']['outside']['events'],1)
 def test_missing_correlation_blocks_denominator(self):
  ev=example_trace()['traceEvents'];next(e for e in ev if e.get('cat')=='kernel')['args'].pop('correlation');d=inspect_events(ev,'lean');self.assertFalse(d['model_denominator_ready']);self.assertEqual(d['unresolved_kernel_count'],1)
 def test_runtime_driver_agree(self):
  ev=example_trace()['traceEvents'];h=copy.deepcopy(next(e for e in ev if e.get('cat')=='cuda_runtime'));h['cat']='cuda_driver';ev.append(h);self.assertTrue(inspect_events(ev,'lean')['model_denominator_ready'])
 def test_conflicting_host_correlation_blocks_denominator(self):
  ev=example_trace()['traceEvents'];h=copy.deepcopy(next(e for e in ev if e.get('cat')=='cuda_runtime'));h.update(cat='cuda_driver',ts=24);ev.append(h);d=inspect_events(ev,'lean');self.assertFalse(d['model_denominator_ready']);self.assertEqual(d['partition']['kernel']['ambiguous']['events'],1)
 def test_missing_scope_holds(self):
  ev=example_trace()['traceEvents'];ev.pop(1)
  with self.assertRaises(AssertionError):inspect_events(ev,'lean')
 def test_duplicate_scope_holds(self):
  ev=example_trace()['traceEvents'];ev.append(copy.deepcopy(ev[1]))
  with self.assertRaises(AssertionError):inspect_events(ev,'lean')
 def test_reordered_or_overlap_scopes_hold(self):
  ev=example_trace()['traceEvents'];ev[2]['ts']=18
  with self.assertRaises(AssertionError):inspect_events(ev,'lean')
 def test_correlated_transfer_is_separate(self):
  ev=example_trace()['traceEvents'];e=copy.deepcopy(next(e for e in ev if e.get('cat')=='kernel'));e['cat']='gpu_memcpy';ev.append(e);d=inspect_events(ev,'lean');self.assertEqual(d['request_kernel_duration_us'],2);self.assertEqual(d['partition']['gpu_memcpy']['execution']['duration_us'],2)
 def test_missing_transfer_correlation_reported_not_hidden(self):
  ev=example_trace()['traceEvents'];ev.append(dict(ph='X',cat='gpu_memcpy',name='copy',pid=0,tid=7,ts=10,dur=2,args={}));d=inspect_events(ev,'lean');self.assertTrue(d['model_denominator_ready']);self.assertEqual(d['partition']['gpu_memcpy']['uncorrelated']['events'],1)
class SourceBoundaryTests(unittest.TestCase):
 def test_timer_numeric_block_identical_to_v1(self):
  old=ast.parse((R.parent/'lean_public_balanced_measure_v1/request.py').read_text());new=ast.parse((R/'request.py').read_text())
  of=next(x for x in old.body if isinstance(x,ast.FunctionDef)and x.name=='request');nf=next(x for x in new.body if isinstance(x,ast.FunctionDef)and x.name=='request')
  def assign(x,name):return isinstance(x,ast.Assign)and isinstance(x.targets[0],ast.Name)and x.targets[0].id==name
  a=next(i for i,x in enumerate(of.body)if assign(x,'begin'));b=next(i for i,x in enumerate(of.body)if assign(x,'end'))
  nw=next(x for x in nf.body if isinstance(x,ast.With)and isinstance(x.items[0].context_expr,ast.Call)and x.items[0].context_expr.args[0].value=='execution')
  self.assertEqual([ast.dump(x)for x in of.body[a:b+1]],[ast.dump(x)for x in nw.body])
 def test_wall_never_profiled(self):
  t=ast.parse((R/'run.py').read_text());calls=[x for x in ast.walk(t)if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='run'];profile=[x for x in calls if any(k.arg=='profiled'for k in x.keywords)]
  self.assertEqual(len(profile),1);self.assertIs(profile[0].args[2].value,False);self.assertIs(next(k.value.value for k in profile[0].keywords if k.arg=='profiled'),True)
 def test_fresh_scope_names(self):self.assertNotEqual(scope_names('native')['execution'],scope_names('lean')['execution'])
 def test_profile_inside_execution_only(self):
  # Exercise real request() with phase-tracking engine/CUDA mocks and output gate mocks.
  current=[];events=[]
  @contextmanager
  def phase(n):current.append(n);events.append(('enter',n));yield;events.append(('exit',current.pop()))
  def record(n):events.append((n,tuple(current)))
  ids=[1]*53;tokens=[[0]*16 for _ in range(32)];keys=[];params=object();model=types.SimpleNamespace()
  class A:
   ATTR='lean'
   def install(self,m):record('install');setattr(m,self.ATTR,1);return{}
   def start(self,m,mode):record('start')
   def finish(self,m):record('census');return{}
   def close(self,m):record('close');return{}
   def release(self,m):record('release');delattr(m,self.ATTR)
  adapter=A()
  class E:
   unfinished=False
   def has_unfinished_requests(self):return self.unfinished
   def apply_model(self,fn):return[fn(model)]
   def add_request(self,k,inp,par):record('add');keys.append(k);self.unfinished=True
   def step(self):
    record('step');self.unfinished=False
    return[types.SimpleNamespace(finished=True,request_id=k,outputs=[types.SimpleNamespace(token_ids=tokens[i])])for i,k in enumerate(keys)]
  cuda=types.SimpleNamespace(synchronize=lambda:record('sync'),reset_peak_memory_stats=lambda:record('reset'),max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)
  torch=types.SimpleNamespace(cuda=cuda,profiler=types.SimpleNamespace(record_function=phase),backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_bf16_reduced_precision_reduction_split_k=True))))
  with patch.object(request,'public_logprobs',lambda *a:[]),patch.object(request,'exact_tokens',lambda *a:{}),patch.object(request,'exact_public',lambda *a:{}),patch.object(request,'dispatch',lambda *a:{}):
   request.request(E(),torch,types.SimpleNamespace(select=lambda _:('native',True)),adapter,{'installed':False},ids,params,'mock','lean',False,tokens,[],lambda _:record('proof_save'),clock=lambda:(_ for _ in()).throw(AssertionError('clock called')),profiled=True)
  for operation,stack in events:
   if operation in('add','step'):self.assertEqual(stack,('request_execution::lean',))
   elif operation in('install','start','reset'):self.assertEqual(stack,('request_install::lean',))
   elif operation=='census':self.assertEqual(stack,('request_proof::lean',))
   elif operation in('close','release'):self.assertEqual(stack,('request_restore::lean',))
  self.assertEqual(current,[])
if __name__=='__main__':unittest.main()
