import ast,copy,contextlib,json,os,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from . import adapter
from .result_gate import audit
from .gates import qualification
from .precision import PrecisionPolicy
R=Path(__file__).parent;P=R.parent
DT=object();DEVICE=types.SimpleNamespace(type='cuda')
class FakeTensor:
 def __init__(self,name,shape,ptr=None,grad=False):self.name=name;self.shape=shape;self.dtype=DT;self.device=DEVICE;self.requires_grad=grad;self.ptr=ptr or id(self)
 def stride(self):return(self.shape[1],1)
 def storage_offset(self):return 0
 def data_ptr(self):return self.ptr
 def untyped_storage(self):return types.SimpleNamespace(data_ptr=lambda:self.ptr,nbytes=lambda:2*self.shape[0]*self.shape[1])
 def detach(self):return FakeTensor(self.name,self.shape,self.ptr,False)
 def cpu(self):return self

def default_unquantized_gemm():pass
class UnquantizedLinearMethod:
 def __init__(self):self._gemm_impl=default_unquantized_gemm
 def apply(self,layer,x,bias=None):layer.native_calls+=1;return FakeTensor('Y', [x.shape[0],64])

def fake_model():
 rows=[]
 for i in adapter.LAYERS:
  q=UnquantizedLinearMethod();m=types.SimpleNamespace(quant_method=q,bias=None,weight=FakeTensor('W'+str(i),[64,2560],grad=True),native_calls=0)
  rows.append((f'language_model.model.layers.{i}.linear_attn.in_proj_ba',m))
 other=types.SimpleNamespace(quant_method=UnquantizedLinearMethod(),native_calls=0)
 return types.SimpleNamespace(named_modules=lambda:rows+[('unrelated.gate_up',other)]),rows,other

def fake_modules():
 mods={}
 for name in ['vllm','vllm.model_executor','vllm.model_executor.layers','vllm.model_executor.layers.linear','vllm.model_executor.layers.utils']:
  mods[name]=types.ModuleType(name);mods[name].__path__=[]
 mods['vllm.model_executor.layers.linear'].UnquantizedLinearMethod=UnquantizedLinearMethod
 mods['vllm.model_executor.layers.utils'].default_unquantized_gemm=default_unquantized_gemm
 mods['torch']=types.SimpleNamespace(Tensor=FakeTensor,bfloat16=DT,no_grad=contextlib.nullcontext,isfinite=lambda t:types.SimpleNamespace(all=lambda:True))
 return mods
GOOD=dict(raw_equal=True,raw_mismatches=0,strict_failures=0,candidate_nonfinite=0,reference_nonfinite=0,elements=2048)

def result_fixture():
 names=[f'language_model.model.layers.{i}.linear_attn.in_proj_ba'for i in adapter.LAYERS]
 def arm(native):
  rows=[]
  for name in names:
   rows.append(dict(path=name,phase='prefill',passed=True,original_calls=1,tileir_calls=0))
   for idx in range(1,16):rows.append(dict(path=name,phase='decode',passed=True,original_calls=1 if native else 0,tileir_calls=1,candidate_path='bf16_k128',inputs_preserved=True,input_backing_preserved=True,output_contract_pass=True,output_finite=True,comparison=GOOD.copy(),same_shadow_inputs=True,same_shadow_output=True))
  return dict(selected_dispatch=dict(shadow_pass=True,supported_calls=360,tileir_gate_calls=360,native_gate_calls=360 if native else 0,prefill_native_calls=24,supported_vendor_fallback_calls=0,unsupported_native_fallback_calls=0,other_apply_bindings_unchanged=True,weights_unchanged=True,first_mismatch=None,pending=False,rows=rows),cross_runtime_tokens_equal=True,allow_bf16_reduced_precision_reduction=True,allow_bf16_reduced_precision_reduction_split_k=True,output_token_ids=[[1]*16 for _ in range(32)])
 return dict(complete=True,engine_shutdown=True,adapter_restore=[dict(restored=True,other_apply_bindings_unchanged=True)],adapter_install=[dict(modules=[dict(path=n,layer=i)for n,i in zip(names,adapter.LAYERS)])],precision_policy=dict(restored=True,restored_split_k=True,policy='unchanged native',initial=dict(bf16=True,split_k=True)),shadow_request=arm(True),replacement_request=arm(False),public_logprob_gate=dict(passed=True,exact_numeric=True,max_abs=0,values=2560))

class Tests(unittest.TestCase):
 def test_no_backend_import(self):self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda.tile',sys.modules)
 def test_exact_layer_census(self):
  r=result_fixture();d=r['shadow_request']['selected_dispatch'];names=[x['path']for x in r['adapter_install'][0]['modules']]
  self.assertTrue(adapter.census(d['rows'],names));self.assertFalse(adapter.census(d['rows'][:-1],names))
 def test_replace_requires_complete_shadow(self):
  with tempfile.TemporaryDirectory()as d:
   a=adapter.Adapter(d+'/a')
   with self.assertRaises(AssertionError):a.start('replace')
 def test_full_mock_shadow_replace_restore(self):
  from packed_k128_partial_v1 import api
  from packed_contiguous_split2_qualification_v1 import run
  model,rows,other=fake_model();orig=other.quant_method.apply
  with tempfile.TemporaryDirectory()as d,patch.dict(sys.modules,fake_modules()),patch.object(adapter,'digest',side_effect=lambda t:t.name+str(t.shape)),patch.object(api,'linear',side_effect=lambda x,w:FakeTensor('Y',[32,64])),patch.object(run,'compare',return_value=GOOD.copy()):
   a=adapter.Adapter(d+'/a');info=a.install(model);self.assertEqual(info['eligible_modules'],24)
   for mode in ['shadow','replace']:
    a.start(mode)
    for _,m in rows:
     for i in range(16):
      x=FakeTensor('X'+str(i),[1696 if i==0 else 32,2560]);m.quant_method.apply(m,x)
    done=a.finish();self.assertEqual(done['tileir_gate_calls'],360);self.assertEqual(done['native_gate_calls'],360 if mode=='shadow'else 0)
   self.assertTrue(all(m.native_calls==17 for _,m in rows));self.assertEqual(other.quant_method.apply,orig)
   a.close();self.assertTrue(all('apply'not in m.quant_method.__dict__ for _,m in rows));self.assertEqual(other.quant_method.apply,orig)
 def test_restore_detects_conflict(self):
  q=UnquantizedLinearMethod();orig=q.apply;bound=lambda:None;q.apply=bound
  with tempfile.TemporaryDirectory()as d:
   a=adapter.Adapter(d+'/a');a.saved=[(q,orig,bound)];q.apply=lambda:None
   with self.assertRaises(AssertionError):a.close()
 def test_positive_and_negative_gate(self):
  self.assertTrue(audit(result_fixture())['passed']);self.assertFalse(audit({})['passed'])
  for key in ['same_shadow_inputs','same_shadow_output','input_backing_preserved','output_finite']:
   r=result_fixture();r['replacement_request']['selected_dispatch']['rows'][1][key]=False;self.assertFalse(audit(r)['passed'],key)
  r=result_fixture();r['shadow_request']['selected_dispatch']['rows'][1]['comparison']['raw_mismatches']=1;self.assertFalse(audit(r)['passed'])
  r=result_fixture();r['replacement_request']['selected_dispatch']['native_gate_calls']=1;self.assertFalse(audit(r)['passed'])
 def test_public_gate_and_flags(self):
  r=result_fixture();r['public_logprob_gate']['exact_numeric']=False;self.assertFalse(audit(r)['passed'])
  t=types.SimpleNamespace(backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_bf16_reduced_precision_reduction=True,allow_bf16_reduced_precision_reduction_split_k=True))))
  p=PrecisionPolicy(t);p.select('native');p.restore();t.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
  with self.assertRaises(AssertionError):p.restore()
 def test_actual_saved_proof(self):self.assertEqual(qualification(R)['result_sha256'],'0e9e3bdb2cabc966c8a587093dc1ae5304c28cb3326b8d4d5a12962e1567f1b4')
 def test_fresh_entry_import(self):
  env={k:v for k,v in os.environ.items()if k!='PYTHONPATH'};env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
  code="import runpy,sys;runpy.run_path(%r,run_name='cpu_entry');assert 'torch'not in sys.modules and 'cuda.tile'not in sys.modules"%str(R/'run.py')
  p=subprocess.run([sys.executable,'-B','-c',code],env=env,cwd='/tmp',capture_output=True,text=True,timeout=15);self.assertEqual(p.returncode,0,p.stderr)
 def test_ast(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
