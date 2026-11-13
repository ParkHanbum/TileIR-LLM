"""Non-numerical CPU mocks: source/argument/dispatch/restore contracts only."""
import ast,contextlib,copy,inspect,json,pathlib,sys,types,unittest
from unittest.mock import patch
from .support import Base,call_phase
from . import adapter as a,model as m,metadata as md
P=pathlib.Path(__file__).resolve().parent.parent;R=pathlib.Path(__file__).resolve().parent
class Ctrl:
 def __init__(self):self.active=True;self.failed=None;self.owner_refs=[]
 def latch(self,key,e):
  if self.failed is None:self.failed=(key,repr(e))
class SupportTests(unittest.TestCase):
 def test_inherited_and_existing_restore(self):
  class T:
   def f(self,x):return x
  x=T();old=x.f;c=Ctrl();b=Base(c,'test',1);seen=[]
  b.bind(x,'f',lambda y:(seen.append(y),y)[1]);self.assertEqual(x.f(4),4);b.close();self.assertEqual(x.f,old);self.assertNotIn('f',vars(x))
  x.f=lambda y:y+1;old=x.f;b.bind(x,'f',lambda y:y+2);b.close();self.assertIs(x.f,old)
 def test_foreign_binding_not_overwritten(self):
  c=Ctrl();b=Base(c,'test',1);x=types.SimpleNamespace(f=lambda:1);b.bind(x,'f',lambda:2);foreign=lambda:3;x.f=foreign
  with self.assertRaises(AssertionError):b.close()
  self.assertIs(x.f,foreign)
 def test_guard_exception_latches_before_second_call(self):
  c=Ctrl();b=Base(c,'test',1);seen=[];x=types.SimpleNamespace(f=lambda:None)
  def bad():seen.append(1);raise ValueError('guard')
  b.bind(x,'f',bad)
  with self.assertRaises(ValueError):x.f()
  with self.assertRaises(AssertionError):x.f()
  self.assertEqual(seen,[1]);b.close()
 def test_phase_shape_guard(self):
  d={};self.assertEqual(call_phase(d,'x',1696),'prefill')
  with self.assertRaises(AssertionError):call_phase(d,'x',32)
  d[('x','prefill')]=1;self.assertEqual(call_phase(d,'x',32),'decode')
class LifecycleTests(unittest.TestCase):
 def exercise(self,fail_install=None,fail_close=None):
  events=[];spec=[]
  for key,cls,count in a.SPECS:
   def install(self,model,_key=key):
    events.append(('install',_key))
    if _key==fail_install:raise ValueError('install '+_key)
    return {}
   def close(self,_key=key):
    events.append(('close',_key))
    if _key==fail_close:raise ValueError('close '+_key)
    return dict(restored=True)
   cls=type('Mock'+key,(Base,),dict(install=install,close=close));spec.append((key,cls,count))
  c=a.Controller(object(),tuple(spec))
  if fail_install:
   with self.assertRaises(ValueError):c.install()
   expected=list(reversed(list(a.ORDER)[:list(a.ORDER).index(fail_install)+1]))
  else:
   c.install()
   if fail_close:
    with self.assertRaises(AssertionError):c.close()
   else:self.assertEqual(c.close()['close_order'],list(reversed(a.ORDER)))
   expected=list(reversed(a.ORDER))
  self.assertEqual([k for t,k in events if t=='close'],expected)
 def test_full_reverse_close(self):self.exercise()
 def test_partial_install_cleanup(self):self.exercise(fail_install='all_gemm')
 def test_close_failure_still_closes_earlier(self):self.exercise(fail_close='all_gemm')
 def test_failed_or_active_cannot_restart(self):
  c=a.Controller(object(),());c.active=True
  with self.assertRaises(AssertionError):c.start()
  c.active=False;c.latch('oom',MemoryError('HOLD'))
  with self.assertRaises(AssertionError):c.start()
class MetadataTests(unittest.TestCase):
 def test_gather_slot_actual_nested_kwargs_identity(self):
  c=Ctrl();g=md.Gather(c,'block_gather',16);s=md.Slots(c,'slot_mapping',16);seen=[];out1=object();out2=object()
  class B:
   def gather_block_tables(self,idx,num_reqs_padded,out=None,out_ptrs=None):raise RuntimeError('native forbidden')
   def compute_slot_mappings(self,idx_mapping,query_start_loc,positions,num_tokens_padded,out=None):raise RuntimeError('native forbidden')
  class Runner:
   def prepare_attn(self,b):
    x=self.block_tables.gather_block_tables(idx=b.idx_mapping,num_reqs_padded=32)
    y=self.block_tables.compute_slot_mappings(idx_mapping=b.idx_mapping,query_start_loc=b.query_start_loc,positions=b.positions,num_tokens_padded=b.num_tokens_after_padding)
    return x,y
  old=Runner.prepare_attn;b=types.SimpleNamespace(idx_mapping=object(),query_start_loc=object(),positions=object(),num_tokens_after_padding=1696);runner=Runner();runner.block_tables=B();runner.pcp_manager=None
  g.apply=lambda owner,idx,padded,metadata:g.call('runner','prefill',lambda:(seen.append(('g',owner,idx,padded)),out1)[1]);s.apply=lambda owner,idx,starts,pos,padded,metadata:s.call('runner','prefill',lambda:(seen.append(('s',owner,idx,starts,pos,padded)),out2)[1])
  with patch.object(md,'phase_metadata',lambda b:dict(num_reqs=32,num_reqs_after_padding=32,phase='prefill')):
   g.install_classes(Runner,B);gw=Runner.prepare_attn;s.install_classes(Runner,B)
   try:self.assertEqual(runner.prepare_attn(b),(out1,out2));self.assertIsNone(g.pending);self.assertIsNone(s.pending);self.assertIs(seen[0][2],b.idx_mapping);self.assertIs(seen[1][3],b.query_start_loc)
   finally:s.close();self.assertIs(Runner.prepare_attn,gw);g.close()
  self.assertIs(Runner.prepare_attn,old)
 def test_positions_keyword_domain_guard_before_api(self):
  class V:
   def __init__(self,v):self.v=v
   def detach(self):return self
   def cpu(self):return self
   def tolist(self):return self.v
   def __len__(self):return len(self.v)
  c=Ctrl();o=md.Positions(c,'pos_seq',16);idx=V([0]);starts=V([0,1]);computed=V([-1]);pos=V([0]);seq=V([0]);calls=[]
  module=types.SimpleNamespace(prepare_pos_seq_lens=lambda *args:None)
  class Runner:
   def prepare_inputs(self,*args):return module.prepare_pos_seq_lens(idx_mapping=idx,query_start_loc=starts,num_computed_tokens=computed,pos=pos,seq_lens=seq)
  runner=Runner();runner.pcp_manager=None;runner.input_buffers=types.SimpleNamespace(positions=pos,seq_lens=seq);runner.req_states=types.SimpleNamespace(num_computed_tokens=types.SimpleNamespace(gpu=computed))
  import pos_seq_runtime_capture_v2.observer as observer
  api=types.SimpleNamespace(prepare_pos_seq_lens=lambda *args:calls.append(args))
  with patch.object(observer,'state_metadata',lambda b:dict(phase='prefill',idx_mapping=[0])),patch.dict(sys.modules,{'tileir_vllm_pos_seq_v1':api}):
   o.install_classes(Runner,module)
   try:
    with self.assertRaises(AssertionError):runner.prepare_inputs(None,None,None,None)
    self.assertEqual(calls,[]);self.assertIsNotNone(c.failed)
   finally:o.close()
class GemmTests(unittest.TestCase):
 def fixture(self):
  from all_gemm_runtime_integration_v1.contract import plan
  import all_gemm_runtime_integration_v1.adapter as old
  p=plan();seen=[]
  class Tensor:
   def __init__(self,shape):self.shape=tuple(shape);self.version=0
   def detach(self):return self
  class Linear:
   def apply(self,owner,x,bias=None):raise RuntimeError('native forbidden')
  class Embed(Linear):pass
  native=lambda *args:None;items=[]
  for spec in p['selected']+p['excluded']:
   kind=spec.get('kind');q=(Embed if kind=='head'else Linear)();q._gemm_impl=native;m=types.SimpleNamespace(quant_method=q,weight=Tensor(spec['weight_shape']),bias=None);items.append((spec['path'],m))
  model=types.SimpleNamespace(named_modules=lambda:items)
  def candidate(tag):
   def fn(x,w):seen.append((tag,x,w));return object()
   return fn
  env=types.SimpleNamespace(VLLM_BATCH_INVARIANT=False)
  stubs={'torch':types.SimpleNamespace(no_grad=contextlib.nullcontext),'vllm':types.SimpleNamespace(envs=env),'vllm.model_executor.layers.linear':types.SimpleNamespace(UnquantizedLinearMethod=Linear),'vllm.model_executor.layers.vocab_parallel_embedding':types.SimpleNamespace(UnquantizedEmbeddingMethod=Embed),'vllm.model_executor.layers.utils':types.SimpleNamespace(default_unquantized_gemm=native,dispatch_unquantized_gemm=lambda:native),'packed_k128_partial_v1':types.SimpleNamespace(linear=candidate('K128')),'tileirblas_once_occ2':types.SimpleNamespace(linear=candidate('existing'))}
  return p,old,Tensor,model,items,stubs,seen
 def exercise(self,wrong=False):
  p,old,T,model,items,stubs,seen=self.fixture();c=Ctrl();g=m.Gemm(c,'all_gemm',2448)
  with patch.dict(sys.modules,stubs),patch.object(old,'dense',lambda t,shape,torch:self.assertEqual(tuple(t.shape),tuple(shape))),patch.object(old,'metadata',lambda t:dict(shape=t.shape,version=t.version,identity=id(t))):
   g.install(model)
   try:
    lookup=dict(items)
    if wrong:
     spec=p['selected'][0];owner=lookup[spec['path']];owner.weight.version=1
     with self.assertRaises(AssertionError):owner.quant_method.apply(owner,T((spec['first_M'],spec['weight_shape'][1])))
     self.assertFalse(seen);self.assertIsNotNone(c.failed)
    else:
     for index in range(16):
      for spec in p['selected']:
       owner=lookup[spec['path']];x=T((spec['first_M']if index==0 else 32,spec['weight_shape'][1]));owner.quant_method.apply(owner,x)
       self.assertIs(seen[-1][1],x);self.assertIs(seen[-1][2],owner.weight)
     r=g.finish();self.assertEqual(r['calls'],2448);self.assertEqual(r['k128_calls'],360);self.assertEqual(r['existing_calls'],2088);self.assertEqual(sum(x[0]=='K128'for x in seen),360)
   finally:g.close()
  self.assertTrue(all('apply'not in vars(owner.quant_method)for name,owner in items))
 def test_all153_2448_dispatch_same_inputs_no_native(self):self.exercise()
 def test_weight_metadata_guard_latches(self):self.exercise(True)
class SourceTests(unittest.TestCase):
 def test_ast_and_no_external_diagnostics(self):
  for name in ['support.py','adapter.py','model.py','metadata.py','recurrent.py','sampler.py']:
   s=(R/name).read_text();ast.parse(s)
   for forbidden in('.synchronize(','.clone(','.write_text(','torch.save(','torch.equal(','hashlib'):
    self.assertNotIn(forbidden,s,(name,forbidden))
 def test_no_numeric_sources_or_launcher(self):
  self.assertFalse((R/'kernels.py').exists());self.assertFalse((R/'launch.py').exists());self.assertFalse((R/'cache_v1').exists())
 def test_import_no_torch_cuda_backend(self):
  import subprocess,os
  code="import sys;sys.path.insert(0,%r);import lean_public_composition_v1.adapter;assert 'torch' not in sys.modules and 'cuda.tile' not in sys.modules and 'vllm' not in sys.modules"%str(P)
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1');r=subprocess.run([sys.executable,'-B','-c',code],env=env,capture_output=True,text=True,timeout=30);self.assertEqual(r.returncode,0,r.stderr)
if __name__=='__main__':unittest.main()
