import ast,copy,json,os,pathlib,subprocess,sys,tempfile,types,unittest
from unittest.mock import patch
from . import adapter as a,result_gate as g
P=pathlib.Path(__file__).resolve().parent.parent;R=pathlib.Path(__file__).resolve().parent
class ControllerTests(unittest.TestCase):
 def setUp(self):
  from . import entry_guards
  self.patch=patch.object(entry_guards,'endpoints',lambda key,obj:[(obj,'mock_entry',obj.mock_entry)])
  self.patch.start();self.addCleanup(self.patch.stop)
 def fake(self,events,fail_install=None,fail_close=None):
  parts=[]
  for key,_,n in a.PARTS:
   m=types.SimpleNamespace(ATTR='_mock_'+key)
   def install(model,out,k=key,m=m):
    events.append(('install',k));obj=types.SimpleNamespace(rows=[],first_mismatch=None,save=lambda:events.append(('save',k)),mock_entry=lambda:None);obj._cpu_mock_endpoint=True;setattr(model,m.ATTR,obj)
    if k==fail_install:raise RuntimeError('install '+k)
    return dict(ok=True)
   def start(model,mode,k=key):events.append(('start',k));return {}
   def finish(model,k=key,n=n):return {'selected_calls'if k=='all_gemm'else'supported_calls':n,'shadow_pass':True,'logical_suboperation_calls':64 if k in ('sampler_four','runner_padding') else (96 if k=='gdn_parent_state' else n),'parent_api_calls':96 if k=='gdn_parent_state' else 0}
   def close(model,k=key,m=m):
    events.append(('close',k))
    if hasattr(model,m.ATTR):delattr(model,m.ATTR)
    if k==fail_close:raise RuntimeError('close '+k)
    return dict(restored=True)
   m.install,m.start,m.finish,m.close=install,start,finish,close;parts.append((key,m,n))
  return parts
 def test_fixed_order_full_lifecycle(self):
  e=[];model=types.SimpleNamespace()
  with tempfile.TemporaryDirectory()as d:
   c=a.Controller(model,d,self.fake(e));v=c.install();self.assertEqual(v['install_order'],list(a.ORDER));c.start('shadow');self.assertTrue(c.finish()['shadow_pass']);c.start('replace');v=c.finish();self.assertEqual(v['selected_logical_operations'],6056);self.assertEqual(v['selected_boundary_calls'],5888);r=c.close();self.assertEqual(r['close_order'],list(reversed(a.ORDER)))
   self.assertEqual([k for t,k in e if t=='close'],list(reversed(a.ORDER)))
 def test_partial_install_all_attempted_cleanup(self):
  e=[];model=types.SimpleNamespace()
  with tempfile.TemporaryDirectory()as d:
   c=a.Controller(model,d,self.fake(e,fail_install='all_gemm'))
   with self.assertRaises(RuntimeError):c.install()
   c.close();self.assertEqual([k for t,k in e if t=='close'],list(reversed(a.ORDER[:a.ORDER.index('all_gemm')+1])));self.assertEqual(model.__dict__,{})
 def test_middle_close_failure_does_not_skip_earlier(self):
  e=[];model=types.SimpleNamespace()
  with tempfile.TemporaryDirectory()as d:
   c=a.Controller(model,d,self.fake(e,fail_close='all_gemm'));c.install()
   with self.assertRaises(AssertionError):c.close()
   self.assertEqual([k for t,k in e if t=='close'],list(reversed(a.ORDER)));self.assertEqual(model.__dict__,{})
 def test_legacy_save_then_failure_latched(self):
  e=[];model=types.SimpleNamespace()
  with tempfile.TemporaryDirectory()as d:
   c=a.Controller(model,d,self.fake(e));c.install();ob=getattr(model,'_mock_norm');ob.rows.append(dict(passed=False));ob.first_mismatch={'stored':'evidence'}
   with self.assertRaises(RuntimeError):ob.save()
   self.assertIn(('save','norm'),e);self.assertIsNotNone(c.failed)
   with self.assertRaises(AssertionError):c.start('shadow')
   c.close();self.assertEqual(model.__dict__,{})
 def test_replace_before_shadow_blocked(self):
  e=[];model=types.SimpleNamespace()
  with tempfile.TemporaryDirectory()as d:
   c=a.Controller(model,d,self.fake(e));c.install()
   with self.assertRaises(AssertionError):c.start('replace')
   c.close()
 def test_oom_start_not_retried_or_settings_changed(self):
  e=[];model=types.SimpleNamespace();parts=self.fake(e)
  def oom(*args):raise MemoryError('simulated capacity HOLD')
  parts[5][1].start=oom
  with tempfile.TemporaryDirectory()as d:
   c=a.Controller(model,d,parts);c.install()
   with self.assertRaises(MemoryError):c.start('shadow')
   with self.assertRaises(AssertionError):c.start('shadow')
   self.assertEqual(sum(t=='start' for t,k in e),5);c.close()
class NestedTests(unittest.TestCase):
 def exercise(self,fail=False):
  from sample_counts_runtime_integration_v1 import adapter as counts
  from greedy_sampler_runtime_integration_v1 import adapter as greedy
  from sampler_four_runtime_integration_v1 import adapter as top
  events=[];sampled=object();number=object();pair=(number,object());output=object()
  def gumbel(logits):events.append('native-greedy');return sampled
  def get_counts(*args):events.append('native-counts');return pair
  def scores(logits):
   events.append('native-scores')
   if fail:raise ValueError('score body failure')
   return output
  mod=types.SimpleNamespace(gumbel_sample=gumbel,get_num_sampled_and_rejected=get_counts,compute_topk_scores=scores)
  batch=types.SimpleNamespace(seq_lens=object(),cu_num_logits=object(),idx_mapping=object())
  class S:
   logprobs_mode='raw_logprobs';trace_replay_state=None
   def __call__(self,logits,b):
    assert mod.gumbel_sample(logits)is sampled
    assert mod.get_num_sampled_and_rejected(number,b.seq_lens,b.cu_num_logits,b.idx_mapping,object())is pair
    return mod.compute_topk_scores(logits)
  original=S.__call__
  with tempfile.TemporaryDirectory()as d:
   cs=counts.Adapter(pathlib.Path(d)/'counts');gr=greedy.Adapter(pathlib.Path(d)/'greedy');tp=top.Adapter(pathlib.Path(d)/'top')
   cs.apply=lambda owner,args,native,metadata:native();gr.apply=lambda vals,native,metadata:native();tp.apply=lambda vals,native,metadata:native()
   with patch.object(counts,'phase_metadata',lambda b:{}),patch.object(greedy,'phase_metadata',lambda b:{}),patch.object(top,'phase_metadata',lambda b:{}):
    cs.install_classes(S,mod);cwrapper=S.__call__;gr.install_classes(S,mod);gwrapper=S.__call__;tp.install_classes(S,mod)
    try:
     if fail:
      with self.assertRaises(ValueError):S()(object(),batch)
     else:self.assertIs(S()(object(),batch),output)
     for obj in (cs,gr,tp):self.assertIsNone(obj.pending)
    finally:
     tp.close();self.assertIs(S.__call__,gwrapper);gr.close();self.assertIs(S.__call__,cwrapper);cs.close()
   self.assertIs(S.__call__,original);self.assertIs(mod.gumbel_sample,gumbel);self.assertIs(mod.compute_topk_scores,scores);self.assertIs(mod.get_num_sampled_and_rejected,get_counts)
   self.assertEqual(events,['native-greedy','native-counts','native-scores'])
 def test_three_actual_wrappers(self):self.exercise()
 def test_three_actual_wrappers_exception(self):self.exercise(True)
class GateTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  # Schema-only composition of separate records. NEVER a runtime proof.
  refs=json.loads((R/'QUALIFICATION-RECEIPT.json').read_text());saved={k:json.loads(pathlib.Path(v['result']['path']).read_text()) for k,v in refs.items()}
  base=saved['base18'];r=copy.deepcopy(base);r['cache_preseed']['destination']=str(R/'cache_v1')
  install=r['adapter_install'][0];del install['components']['sampler_three'];install['install_order']=list(a.ORDER);install.update(controller_count=25,selected_boundary_calls_expected=5888,selected_logical_operations_expected=6056,entry_guard_map={k:[dict(kind='installed_binding',attribute='mock')] for k in a.ORDER})
  restore=r['adapter_restore'][0];del restore['components']['sampler_three'];restore['close_order']=list(reversed(a.ORDER));restore['entry_guards_restored']=True
  for key in ('sampler_four','gdn_prefill_core','attention_core','runner_padding','hidden_gather','sampler_fp32','accepted_scatter'):
   install['components'][key]=saved[key]['adapter_install'][0];restore['components'][key]=saved[key]['adapter_restore'][0]
  for phase in ('shadow_request','replacement_request'):
   d=r[phase]['selected_dispatch'];del d['components']['sampler_three']
   for key in ('sampler_four','gdn_prefill_core','attention_core','runner_padding','hidden_gather','sampler_fp32','accepted_scatter'):d['components'][key]=saved[key][phase]['selected_dispatch']
   d.update(selected_boundary_calls=5888,selected_logical_operations=6056,installation_order=list(a.ORDER))
  install['components']['gdn_parent_state']=dict(core_delegated=True,core_native=False)
  restore['components']['gdn_parent_state']=dict(restored=True,core_guard_restored=True)
  for phase,mode in (('shadow_request','shadow'),('replacement_request','replace')):
   d=r[phase]['selected_dispatch'];core=d['components']['gdn_prefill_core'];parents=[x['path'].rsplit('.',1)[0] for x in core['rows']]
   rows=[dict(layer=p,passed=True,order=['gather','invert','mask','core_enter','core_exit','scatter'],native_target_calls=3 if mode=='shadow' else 0,tile_target_calls=0 if mode=='shadow' else 3,native_invert_calls=1 if mode=='shadow' else 0,tile_invert_calls=0 if mode=='shadow' else 1) for p in parents]
   d['components']['gdn_parent_state']=dict(mode=mode,rows=rows,failed=None,native_only=False,parent_targets_native=mode=='shadow',owner_payload_bytes=0,metadata_only=False,core_native=False,core_delegated=True,native_invert_retained=mode=='shadow',target_oracles=0,supported_calls=24,logical_suboperation_calls=96,parent_api_calls=96,supported_vendor_fallback_calls=0,shadow_pass=True,core_links=[dict(parent=p,core_path=p+'.chunk_gated_delta_rule',exact_installed_guard=True,captured_by_parent=True) for p in parents],core_phase_rows=[dict(path=x['path'],phase=x['phase'],passed=x['passed'])for x in core['rows']],core_native_calls=core['native_gate_calls'],core_tileir_calls=core['tileir_gate_calls'])
  cls.example=r
 def test_schema_only_projection_accepts(self):self.assertTrue(g.audit(self.example)['passed'])
 def test_missing_component_rejected(self):
  r=copy.deepcopy(self.example);del r['replacement_request']['selected_dispatch']['components']['all_gemm'];self.assertFalse(g.audit(r)['passed'])
 def test_w_boundary_drift_rejected(self):
  r=copy.deepcopy(self.example);r['replacement_request']['selected_dispatch']['components']['all_gemm']['weight_boundaries']['replace_end']['records']['weight0']['sha256']='wrong';self.assertFalse(g.audit(r)['passed'])
 def test_top5_score_failure_rejected(self):
  r=copy.deepcopy(self.example);r['shadow_request']['selected_dispatch']['components']['sampler_four']['rows'][0]['passed']=False;self.assertFalse(g.audit(r)['passed'])
 def test_stale_cache_destination_rejected(self):
  r=copy.deepcopy(self.example);r['cache_preseed']['destination']=str(P/'old/cache_v1');self.assertFalse(g.audit(r)['passed'])
 def test_wrong_close_order_rejected(self):
  r=copy.deepcopy(self.example);r['adapter_restore'][0]['close_order']=list(a.ORDER);self.assertFalse(g.audit(r)['passed'])
class SourceTests(unittest.TestCase):
 def test_ast(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
 def test_no_old_receipt_or_cache(self):
  self.assertFalse((R/'CACHE-PRESEED-01.json').exists());self.assertFalse((R/'cache_v1').exists())
 def test_no_added_candidate_paths(self):
  self.assertEqual(tuple(k for k,m,n in a.SIXTEEN),a.ORDER[:16]);self.assertEqual(a.ORDER[16:],('all_gemm','sampler_four','gdn_prefill_core','attention_core','runner_padding','hidden_gather','sampler_fp32','accepted_scatter','gdn_parent_state'))
 def test_actual_entry_import_no_torch(self):
  s="import runpy,sys; sys.argv=['run.py']; runpy.run_path(%r,run_name='__main__'); assert 'torch' not in sys.modules and 'vllm' not in sys.modules"%str(R/'run.py')
  env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='');env.pop('PYTHONPATH',None)
  p=subprocess.run([sys.executable,'-B','-c',s],env=env,capture_output=True,text=True,timeout=20);self.assertEqual(p.returncode,0,p.stderr)
if __name__=='__main__':unittest.main()
