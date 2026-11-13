import ast,json,unittest,types,inspect,textwrap,hashlib,tempfile
from pathlib import Path
from .request_cleanup import finish_observation
P=Path(__file__).resolve().parent.parent
class Lifetime(unittest.TestCase):
 def test_actual_owned_prepare_body_reaches_wrapped_aliases(self):
  # Compile the exact installed public method and actual owned_copy implementation;
  # all data/runtime dependencies below are stdlib mocks, no Torch/vLLM import.
  folder=P/'runner_padding_runtime_integration_v1';spec=json.loads((folder/'SOURCE.json').read_text())
  tree=ast.parse(Path(spec['path']).read_text());node=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='prepare_inputs' and any(isinstance(x,ast.Name)and x.id=='combine_sampled_and_draft_tokens'for x in ast.walk(n)))
  calls=[];token=object()
  class Done(Exception):pass
  def prefill(*a):calls.append('native_prefill');return None
  def combine(*a):calls.append('native_combine');return token
  def wrap(fn,label):
   def f(*a):
    calls.append(label);r=fn(*a)
    if label=='combine':self.assertIs(r,token);raise Done()
    self.assertIsNone(r);return r
   return f
  class A:
   def __getitem__(self,k):return self
   def __setitem__(self,k,v):pass
  arr=A();ns=dict(prepare_prefill_inputs=prefill,combine_sampled_and_draft_tokens=combine,envs=types.SimpleNamespace(VLLM_MOE_SKIP_PADDING=False),async_tensor_h2d=lambda *a,**k:arr,torch=types.SimpleNamespace(int64='int64'),np=types.SimpleNamespace(int32='int32',arange=lambda *a,**k:arr,empty=lambda *a,**k:arr,cumsum=lambda *a,**k:None),prepare_pos_seq_lens=lambda *a:None)
  module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[]));exec(compile(module,spec['path'],'exec'),ns);original=ns['prepare_inputs']
  ns['prepare_prefill_inputs']=wrap(prefill,'prefill');ns['combine_sampled_and_draft_tokens']=wrap(combine,'combine')
  ownnode=next(n for n in ast.parse((P/'lean_public_composition_v2/attention_runner.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='owned_copy')
  ownenv=dict(P=P,inspect=inspect,json=json,hashlib=hashlib,Path=Path,textwrap=textwrap)
  exec(compile(ast.Module(body=[ownnode],type_ignores=[]),'<actual owned_copy>','exec'),ownenv)
  copied=ownenv['owned_copy'](original,'runner_padding_runtime_integration_v1','prepare_inputs',{'_construct':lambda *a:arr,'_fill_padding':lambda *a:None})
  self.assertIs(copied.__globals__['combine_sampled_and_draft_tokens'],ns['combine_sampled_and_draft_tokens'])
  obj=types.SimpleNamespace(input_buffers=types.SimpleNamespace(is_padding=arr,query_start_loc=arr,input_ids=arr,positions=arr,seq_lens=arr),device='mock',model_state=types.SimpleNamespace(num_new_sampled_tokens_per_step=1),max_num_reqs=32,req_states=types.SimpleNamespace(next_prefill_tokens=arr,all_token_ids=types.SimpleNamespace(gpu=arr),prefill_len=types.SimpleNamespace(gpu=arr),num_computed_tokens=types.SimpleNamespace(gpu=arr),last_sampled_tokens=arr,draft_tokens=arr))
  batch=types.SimpleNamespace(num_tokens=1696,req_ids=list(range(32)),num_scheduled_tokens=arr,idx_mapping_np=arr,has_prefill=True)
  try:
   with self.assertRaises(Done):copied(obj,types.SimpleNamespace(scheduled_spec_decode_tokens={}),batch,types.SimpleNamespace(num_tokens=1696,num_reqs=32),0)
  finally:
   ns['prepare_prefill_inputs']=prefill;ns['combine_sampled_and_draft_tokens']=combine
  self.assertEqual(calls,['prefill','native_prefill','combine','native_combine']);self.assertIs(ns['combine_sampled_and_draft_tokens'],combine)
 def cleanup_case(self,fail):
  order=[];result={'complete':True};data={'helpers_restored':False,'failed':False}
  class H:
   def __exit__(self,*a):
    order.append('helpers');data['helpers_restored']=True
    if fail=='helper':raise ValueError('helper failed')
  def cleanup():
   order.append('adapter')
   if fail=='adapter':raise ValueError('adapter failed')
  with tempfile.TemporaryDirectory()as d:
   action=lambda:finish_observation(result,lambda:order.append('save'),cleanup,H(),types.SimpleNamespace(snapshot=lambda:data),Path(d))
   if fail:
    with self.assertRaises(RuntimeError):action()
    self.assertFalse(result['complete'])
   else:action();self.assertTrue(result['complete'])
  self.assertEqual(order,['adapter','helpers','save'])
 def test_cleanup_success(self):self.cleanup_case(None)
 def test_cleanup_adapter_failure(self):self.cleanup_case('adapter')
 def test_cleanup_helper_failure(self):self.cleanup_case('helper')
