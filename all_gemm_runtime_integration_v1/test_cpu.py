import ast,contextlib,copy,hashlib,json,math,os,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from . import adapter
from .contract import plan,phase_shape,coverage
from .result_gate import audit
from .gates import qualification
R=Path(__file__).parent;P=R.parent;DT=object();DEVICE=types.SimpleNamespace(type='cuda')
class Tensor:
 def __init__(self,name,shape,ptr=None,grad=False):self.name=name;self.shape=shape;self.ptr=ptr or id(self);self.dtype=DT;self.device=DEVICE;self.requires_grad=grad;self.version_available=True
 def stride(self):return (self.shape[1],1)
 def storage_offset(self):return 0
 def data_ptr(self):return self.ptr
 def untyped_storage(self):return types.SimpleNamespace(data_ptr=lambda:self.ptr,nbytes=lambda:2*math.prod(self.shape))
 def numel(self):return math.prod(self.shape)
 def element_size(self):return 2
 def detach(self):
  t=Tensor(self.name,self.shape,self.ptr);t.version_available=self.version_available;return t
 @property
 def _version(self):
  if not self.version_available:raise RuntimeError('Inference tensors do not track version counter')
  return 0

def default_gemm():pass
def output(owner,x):
 owner.native_calls+=1
 if owner.fail:raise RuntimeError('injected native failure')
 return Tensor('Y:'+x.name+':'+owner.weight.name,[x.shape[0],owner.weight.shape[0]])
class Linear:
 def __init__(self):self._gemm_impl=default_gemm
 def apply(self,layer,x,bias=None):return output(layer,x)
class Head:
 def apply(self,layer,x,bias=None):return output(layer,x)
Linear.apply.__module__='vllm.model_executor.layers.linear';Linear.apply.__qualname__='UnquantizedLinearMethod.apply'
Head.apply.__module__='vllm.model_executor.layers.vocab_parallel_embedding';Head.apply.__qualname__='UnquantizedEmbeddingMethod.apply'

def fake_modules():
 mods={}
 for name in ['vllm','vllm.model_executor','vllm.model_executor.layers','vllm.model_executor.layers.linear','vllm.model_executor.layers.vocab_parallel_embedding','vllm.model_executor.layers.utils','vllm.envs']:
  mods[name]=types.ModuleType(name);mods[name].__path__=[]
 mods['vllm.model_executor.layers.linear'].UnquantizedLinearMethod=Linear
 mods['vllm.model_executor.layers.vocab_parallel_embedding'].UnquantizedEmbeddingMethod=Head
 mods['vllm.model_executor.layers.utils'].default_unquantized_gemm=default_gemm
 mods['vllm.model_executor.layers.utils'].dispatch_unquantized_gemm=lambda:default_gemm
 mods['vllm.envs'].VLLM_BATCH_INVARIANT=False;mods['vllm'].envs=mods['vllm.envs']
 mods['torch']=types.SimpleNamespace(Tensor=Tensor,bfloat16=DT,no_grad=contextlib.nullcontext)
 return mods

def fake_model():
 p=plan();rows=[]
 for spec in p['selected']+p['excluded']:
  n=spec['path'];q=Head()if n.endswith(('lm_head','embed_tokens'))else Linear()
  rows.append((n,types.SimpleNamespace(quant_method=q,weight=Tensor('W:'+n,spec['weight_shape'],grad=True),bias=None,native_calls=0,fail=False)))
 return types.SimpleNamespace(named_modules=lambda:rows),dict(rows)

def digest(t,torch):return dict(sha256=hashlib.sha256((t.name+str(t.shape)).encode()).hexdigest(),finite=True,elements=t.numel(),bytes=t.numel()*2)
def compare(a,b,torch):
 ah,bh=digest(a,torch)['sha256'],digest(b,torch)['sha256'];same=ah==bh
 return dict(raw_equal=same,raw_mismatches=0 if same else 1,strict_failures=0 if same else 1,reference_nonfinite=0,candidate_nonfinite=0,elements=a.numel(),native_sha256=ah,tile_sha256=bh)
def candidate(x,w):return Tensor('Y:'+x.name+':'+w.name,[x.shape[0],w.shape[0]])

class Tests(unittest.TestCase):
 def test_plan_counts(self):
  p=plan();self.assertEqual(sum(x['calls']for x in p['selected']),2448);self.assertEqual(sum(x['new_K128_calls']for x in p['selected']),360);self.assertEqual(len(p['excluded']),25)
 def test_head_phase_is_index(self):
  spec=next(x for x in plan()['selected']if x['kind']=='head');self.assertEqual(phase_shape(spec,0),('prefill',[32,248320,2560]));self.assertEqual(phase_shape(spec,1),('decode',[32,248320,2560]))
 def test_version_unavailable_explicit(self):
  t=Tensor('t',[1,2]);t.version_available=False;self.assertEqual(adapter.metadata(t)['version'],'unavailable_inference_tensor')
 def test_full_mock_lifecycle_and_gate(self):
  import tileirblas_once_occ2,packed_k128_partial_v1
  model,owners=fake_model();p=plan()
  with tempfile.TemporaryDirectory()as d,patch.dict(sys.modules,fake_modules()),patch.object(adapter,'digest',side_effect=digest),patch.object(adapter,'compare',side_effect=compare),patch.object(tileirblas_once_occ2,'linear',side_effect=candidate),patch.object(packed_k128_partial_v1,'linear',side_effect=candidate):
   a=adapter.Adapter(d+'/a');install=a.install(model);arms=[]
   with patch.object(a,'save'):
    for mode in ('shadow','replace'):
     a.start(mode)
     for idx in range(16):
      for spec in p['selected']:
       name=spec['path'];m=owners[name];_,(M,N,K)=phase_shape(spec,idx);x=Tensor(name+':'+str(idx),[M,K]);m.quant_method.apply(m,x)
     s=a.finish();self.assertTrue(coverage(s['rows'],p));self.assertEqual(s['k128_calls'],360);self.assertEqual(s['existing_calls'],2088);arms.append(dict(selected_dispatch=s,cross_runtime_tokens_equal=True,output_token_ids=[[1]*16 for _ in range(32)],allow_bf16_reduced_precision_reduction=True,allow_bf16_reduced_precision_reduction_split_k=True))
    restored=a.close()
   self.assertTrue(all('apply'not in m.quant_method.__dict__ for m in owners.values()))
   self.assertEqual(sum(m.native_calls for m in owners.values()),2448)
   r=dict(cache_preseed=dict(destination=str(R/'cache_v1'),files_verified=True,separate_inodes_verified=True,file_count=1033),complete=True,engine_shutdown=True,adapter_restore=[restored],adapter_install=[install],precision_policy=dict(policy='unchanged native',initial=dict(bf16=True,split_k=True),restored=True,restored_split_k=True),shadow_request=arms[0],replacement_request=arms[1],public_logprob_gate=dict(passed=True,exact_numeric=True,max_abs=0,values=2560))
   self.assertTrue(audit(r)['passed'])
   for bad in ('hash','weight','count','fallback','nonfinite','flags'):
    z=copy.deepcopy(r)
    if bad=='hash':z['replacement_request']['selected_dispatch']['rows'][0]['output_sha256']='bad'
    if bad=='weight':next(iter(z['replacement_request']['selected_dispatch']['weight_boundaries']['replace_end']['records'].values()))['sha256']='bad'
    if bad=='count':z['shadow_request']['selected_dispatch']['rows'].pop()
    if bad=='fallback':z['replacement_request']['selected_dispatch']['selected_vendor_fallback_calls']=1
    if bad=='nonfinite':z['shadow_request']['selected_dispatch']['rows'][0]['comparison']['candidate_nonfinite']=1
    if bad=='flags':z['replacement_request']['allow_bf16_reduced_precision_reduction']=False
    self.assertFalse(audit(z)['passed'],bad)
 def test_first_failure_latches_all_further_entry(self):
  model,owners=fake_model();p=plan();name=p['selected'][0]['path'];m=owners[name]
  with tempfile.TemporaryDirectory()as d,patch.dict(sys.modules,fake_modules()),patch.object(adapter,'digest',side_effect=digest):
   a=adapter.Adapter(d+'/a');a.install(model)
   with patch.object(a,'save'):
    a.start('shadow');m.fail=True;M,N,K=phase_shape(p['selected'][0],0)[1]
    with self.assertRaises(RuntimeError):m.quant_method.apply(m,Tensor('x',[M,K]))
    self.assertEqual(m.native_calls,1);self.assertIsNotNone(a.failed)
    m.fail=False
    with self.assertRaises(RuntimeError):m.quant_method.apply(m,Tensor('x',[M,K]))
    self.assertEqual(m.native_calls,1)
    a.close()
 def test_numeric_failure_latches_before_next_owner(self):
  import tileirblas_once_occ2
  model,owners=fake_model();p=plan();one,two=p['selected'][:2];m=owners[one['path']];other=owners[two['path']]
  def bad(a,b,t):
   c=compare(a,b,t);c.update(raw_equal=False,raw_mismatches=1,strict_failures=1);return c
  with tempfile.TemporaryDirectory()as d,patch.dict(sys.modules,fake_modules()),patch.object(adapter,'digest',side_effect=digest),patch.object(adapter,'compare',side_effect=bad),patch.object(tileirblas_once_occ2,'linear',side_effect=candidate):
   a=adapter.Adapter(d+'/a');a.install(model)
   with patch.object(a,'save'),patch.object(a,'save_failure')as saved:
    a.start('shadow');M,N,K=phase_shape(one,0)[1]
    with self.assertRaises(RuntimeError):m.quant_method.apply(m,Tensor('x',[M,K]))
    saved.assert_called_once();self.assertEqual(a.tile_calls,1)
    M,N,K=phase_shape(two,0)[1]
    with self.assertRaises(RuntimeError):other.quant_method.apply(other,Tensor('y',[M,K]))
    self.assertEqual(other.native_calls,0);self.assertEqual(a.tile_calls,1);a.close()
 def test_weight_boundary_catches_bytes_with_unchanged_metadata(self):
  model,owners=fake_model();first=plan()['selected'][0]['path']
  with tempfile.TemporaryDirectory()as d,patch.dict(sys.modules,fake_modules()),patch.object(adapter,'digest',side_effect=digest):
   a=adapter.Adapter(d+'/a');a.install(model)
   with patch.object(a,'save'):
    a.start('shadow');a.phase_active=False;a.shadow_pass=True
    w=owners[first].weight;meta=adapter.metadata(w);w.name+='bytechange';self.assertEqual(meta,adapter.metadata(w))
    with self.assertRaises(AssertionError):a.start('replace')
    self.assertIsNotNone(a.failed);a.close()
 def test_excluded_apply_holds_without_native(self):
  model,owners=fake_model();p=plan();m=owners[p['excluded'][0]['path']]
  with tempfile.TemporaryDirectory()as d,patch.dict(sys.modules,fake_modules()),patch.object(adapter,'digest',side_effect=digest):
   a=adapter.Adapter(d+'/a');a.install(model)
   with patch.object(a,'save'):
    a.start('shadow')
    with self.assertRaises(RuntimeError):m.quant_method.apply(m,Tensor('x',[32,2560]))
    self.assertEqual(m.native_calls,0);self.assertIsNotNone(a.failed);a.close()
 def test_phase_finish_failure_latches(self):
  with tempfile.TemporaryDirectory()as d:
   a=adapter.Adapter(d+'/a');a.phase_active=True;a.mode='shadow'
   with patch.object(a,'save'):
    with self.assertRaises(AssertionError):a.finish()
   self.assertIsNotNone(a.failed)
 def test_proof_and_import(self):
  self.assertIn('evidence',qualification(R));self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda.tile',sys.modules)
 def test_fresh_entry_import(self):
  env={k:v for k,v in os.environ.items()if k!='PYTHONPATH'};env.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
  code="import runpy,sys;runpy.run_path(%r,run_name='cpu_entry');assert 'torch'not in sys.modules and 'cuda.tile'not in sys.modules"%str(R/'run.py')
  r=subprocess.run([sys.executable,'-B','-c',code],cwd='/tmp',env=env,text=True,capture_output=True,timeout=15);self.assertEqual(r.returncode,0,r.stderr)
 def test_cache_wrong_destination_and_changed_bytes_hold(self):
  from .cache_gate import preflight
  with tempfile.TemporaryDirectory()as d:
   root=Path(d)/'run';root.mkdir();src=Path(d)/'source';src.mkdir();dest=root/'cache_v1';dest.mkdir();(src/'a').write_bytes(b'abc');(dest/'a').write_bytes(b'abc')
   receipt=dict(source=str(src),destination=str(dest),source_files={'a':hashlib.sha256(b'abc').hexdigest()},count=1,bytes=3,same_bytes=True,distinct_inodes=True,source_unchanged=True)
   path=root/'CACHE-PRESEED-01.json';path.write_text(json.dumps(receipt));self.assertTrue(preflight(root)['files_verified'])
   wrong=dict(receipt,destination=str(src));path.write_text(json.dumps(wrong))
   with self.assertRaises(AssertionError):preflight(root)
   path.write_text(json.dumps(receipt));(dest/'a').write_bytes(b'abd')
   with self.assertRaises(AssertionError):preflight(root)
 def test_cache_missing_receipt_holds(self):
  from .cache_gate import preflight
  with tempfile.TemporaryDirectory()as d:
   with self.assertRaises(AssertionError):preflight(d)
 def test_ast(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
