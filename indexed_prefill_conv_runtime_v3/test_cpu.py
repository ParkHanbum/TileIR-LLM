import unittest,copy,ast,inspect
from pathlib import Path
from .adapter import remap_slots,extent,Adapter
from .result_gate import audit
from .cleanup import final_cleanup
class Tests(unittest.TestCase):
 def test_actual_mapping(self):
  slots=list(range(128,0,-4));selected,mapped=remap_slots(slots,311)
  self.assertEqual(selected,list(range(4,129,4)));self.assertEqual(mapped,list(range(32,0,-1)))
 def test_bad_slots(self):
  for slots in ([1]*32,list(range(32)),list(range(1,32))+[311],[True]+list(range(1,32))):
   with self.assertRaises(AssertionError):remap_slots(slots,311)
 def test_scratch_bounds(self):
  self.assertLess(extent((33,8192,3),(1081344,1,8192))*2,80*1024*1024)
  self.assertLess(extent((8192,1696),(1,12288))*2,64*1024*1024)
 def fixture(self):
  d=dict(shadow_pass=True,supported_calls=24,supported_vendor_fallback_calls=0,first_mismatch=None,rows=[dict(passed=True,original_calls=1,output_finite=True)for _ in range(24)],counts={'x:prefill':24},update_binding_is_install_time_object=True)
  b=copy.deepcopy(d)
  for row in b['rows']:row['original_calls']=0
  return dict(complete=True,engine_shutdown=True,adapter_restore=[{'restored':True,'update_binding_is_install_time_object':True}],public_logprob_gate={'passed':True,'exact_numeric':True,'max_abs':0.0,'values':2560},precision_policy={'restored':True,'restored_split_k':True,'changes':[],'initial_allow_bf16_reduced_precision_reduction':True,'initial_allow_bf16_reduced_precision_reduction_split_k':True},shadow_request={'output_token_ids':[[1]*16]*32,'generated_tokens':512,'selected_dispatch':d},replacement_request={'output_token_ids':[[1]*16]*32,'generated_tokens':512,'selected_dispatch':b})
 def test_gate_positive(self):self.assertTrue(audit(self.fixture())['passed'])
 def test_gate_raw_and_count(self):
  for key,value in [('exact_numeric',False),('max_abs',1e-8),('values',2559)]:
   r=self.fixture();r['public_logprob_gate'][key]=value;self.assertFalse(audit(r)['passed'])
 def test_gate_badrow(self):
  r=self.fixture();r['shadow_request']['selected_dispatch']['rows'][-1]['passed']=False;self.assertFalse(audit(r)['passed'])
 def test_nonfinite_replacement_hold(self):
  r=self.fixture();r['replacement_request']['selected_dispatch']['rows'][0]['output_finite']=False;self.assertFalse(audit(r)['passed'])
 def test_gate_fallback(self):
  r=self.fixture();r['replacement_request']['selected_dispatch']['supported_vendor_fallback_calls']=1;self.assertFalse(audit(r)['passed'])
 def test_gate_update_change(self):
  r=self.fixture();r['replacement_request']['selected_dispatch']['update_binding_is_install_time_object']=False;self.assertFalse(audit(r)['passed'])
 def test_gate_restore_update(self):
  r=self.fixture();r['adapter_restore'][0]['update_binding_is_install_time_object']=False;self.assertFalse(audit(r)['passed'])
 def test_fn_only_source(self):
  tree=ast.parse(Path(inspect.getsourcefile(Adapter)).read_text())
  loops=[n for n in ast.walk(tree)if isinstance(n,ast.For)and isinstance(n.iter,ast.List)]
  self.assertIn([('causal_conv1d_fn','prefill')],[ast.literal_eval(n.iter)for n in loops])
  calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='_orig'];self.assertEqual(len(calls),1)
 def test_finally_ast(self):
  tree=ast.parse(Path(__file__).with_name('run.py').read_text());calls=[]
  for n in ast.walk(tree):
   if isinstance(n,ast.Try):
    for f in n.finalbody:calls.extend(x for x in ast.walk(f)if isinstance(x,ast.Call)and isinstance(x.func,ast.Name)and x.func.id=='final_cleanup')
  self.assertEqual(len(calls),1)
 def test_cleanup_each_failure(self):
  from types import SimpleNamespace
  names=['precision','sync','close','shutdown','alarm','save']
  for fail in names+[None]:
   seen=[];r={'complete':True}
   def action(name,value=None):
    seen.append(name)
    if name==fail:raise RuntimeError(name)
    return value
   policy=SimpleNamespace(restore=lambda:action('precision'));adapter=SimpleNamespace(close=lambda _:action('close',{'restored':True}));engine=SimpleNamespace(apply_model=lambda fn:[fn(None)],engine_core=SimpleNamespace(shutdown=lambda:action('shutdown')))
   final_cleanup(r,lambda:action('save'),policy,engine,adapter,True,False,lambda:action('sync'),lambda:action('alarm'),propagating=True)
   self.assertEqual(seen,names)
   if fail:self.assertFalse(r['complete'])
 def test_cleanup_engine_none(self):
  seen=[];from types import SimpleNamespace
  final_cleanup({},lambda:seen.append('save'),SimpleNamespace(restore=lambda:seen.append('policy')),None,None,True,False,lambda:seen.append('sync'),lambda:seen.append('alarm'),propagating=False)
  self.assertEqual(seen,['policy','alarm','save'])
if __name__=='__main__':unittest.main()
