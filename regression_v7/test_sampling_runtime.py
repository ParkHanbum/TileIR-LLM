"""Draft stdlib contract/result-gate tests; no framework/GPU imports."""
import ast,copy,json,hashlib,importlib.util,pathlib,sys,types,unittest
from . import modules_sampling_runtime as m
path=pathlib.Path(m.__file__).resolve()
class Tests(unittest.TestCase):
 def good(self):return dict(complete=True,records=[dict(family=f,case=c,origin=o,output_descriptor=m.expected_descriptor(f,c),expected_descriptor=m.expected_descriptor(f,c),**{k:True for k in m.FIELDS})for f,c,o in m.expected_rows()])
 def test_exact_count_order(self):
  r=self.good();self.assertEqual(len(r['records']),16);self.assertTrue(m.audit(r));r['records'].reverse();self.assertFalse(m.audit(r))
 def test_incomplete_or_missing(self):
  r=self.good();r['complete']=False;self.assertFalse(m.audit(r));r=self.good();r['records'].pop();self.assertFalse(m.audit(r))
 def test_each_gate_is_required(self):
  for field in m.FIELDS:
   for val in (False,1,None):
    r=self.good();r['records'][0][field]=val;self.assertFalse(m.audit(r))
 def test_no_saved_vs_synthetic_promotion(self):
  r=self.good();r['records'][-1]['origin']='saved_actual';self.assertFalse(m.audit(r))
 def test_unknown_case(self):
  r=self.good();r['records'][0]['case']='16';self.assertFalse(m.audit(r))
 def test_import_is_cpu_only(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda.tile',sys.modules)
 def test_actual_metadata_errors(self):
  from runtime_metadata_five_v1.api import validate_full
  from types import SimpleNamespace
  d=SimpleNamespace(type='cuda')
  validate_full((2048,),-1,dtype='torch.int32',device=d)
  for size,fill,dtype in [((2047,),-1,'torch.int32'),((2048,),True,'torch.int32'),((2048,),-1,'torch.int64')]:
   with self.assertRaises(ValueError):validate_full(size,fill,dtype=dtype,device=d)
 def test_actual_widen_errors(self):
  from tileir_vllm_embedding_int64_v1.contract import validate
  args=[(32,),(1,),'torch.int32',0,'cuda','torch.strided',False,False,False]
  self.assertEqual(validate(*args),32)
  for i,v in ((0,(31,)),(2,'torch.int64'),(3,1),(4,'cpu'),(7,True)):
   a=args.copy();a[i]=v
   with self.assertRaises(AssertionError):validate(*a)
class ActualHelperTests(unittest.TestCase):
 def helper(self,name,env):
  tree=ast.parse(path.read_text());node=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name==name)
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env);return env[name]
 def test_descriptor_negative(self):
  r=Tests().good();r['records'][0]['output_descriptor']['numel']=31;self.assertFalse(m.audit(r))
 def test_load_actual_helper(self):
  calls=[];tensor=object();payload=b'abc'
  fn=self.helper('load',dict(torch=types.SimpleNamespace(load=lambda *a,**kw:(calls.append((a,kw))or tensor)),hashlib=hashlib,raw=lambda x:payload))
  item=dict(path='fixture.pt',logical_sha256=hashlib.sha256(payload).hexdigest())
  self.assertIs(fn(item),tensor);self.assertEqual(calls[0][1],dict(weights_only=True,map_location='cpu'))
  with self.assertRaises(AssertionError):fn(dict(item,logical_sha256='wrong'))
 def test_native_ranks_actual_helper(self):
  calls=[];out=object();x=types.SimpleNamespace(device='cuda:0',stride=lambda i:248320);chosen=object()
  class Kernel:
   def __getitem__(self,grid):
    self.grid=grid
    return lambda *a,**kw:calls.append((grid,a,kw))
  fn=self.helper('native_ranks',dict(torch=types.SimpleNamespace(int64='int64',empty=lambda *a,**kw:out),x=x,chosen=chosen,_ranks_kernel=Kernel()))
  self.assertIs(fn(),out);self.assertEqual(calls,[((32,),(out,x,248320,chosen,248320),dict(BLOCK_SIZE=8192))])
 def test_inspect_actual_helper_success_and_native_mutation(self):
  class T:
   shape=(32,);dtype='torch.int64'
   def __init__(self,ptr,data):self.ptr=ptr;self.data=data
   def stride(self):return (1,)
   def numel(self):return 32
   def storage_offset(self):return 0
   def untyped_storage(self):return types.SimpleNamespace(data_ptr=lambda:self.ptr)
  expected=T(1,b'good');owner=T(2,b'input');candidate=T(3,b'good');result=dict(records=[]);events=[]
  torch=types.SimpleNamespace(cuda=types.SimpleNamespace(current_stream=lambda:types.SimpleNamespace(cuda_stream=7),synchronize=lambda:events.append('sync')),isfinite=lambda x:types.SimpleNamespace(all=lambda:True))
  env=dict(torch=torch,hashlib=hashlib,raw=lambda x:x.data,descriptor=m.descriptor,expected_descriptor=m.expected_descriptor,FIELDS=m.FIELDS,result=result,save=lambda:events.append('save'))
  fn=self.helper('inspect',env);fn('greedy','0','saved_actual',lambda:candidate,expected,[owner],native=lambda:expected)
  self.assertTrue(result['records'][0]['passed']);self.assertEqual(events,['sync','sync','save'])
  def bad():owner.data=b'changed';return expected
  with self.assertRaises(AssertionError):fn('greedy','0','saved_actual',lambda:self.fail('candidate must not run'),expected,[owner],native=bad)
 def test_public_native_signature_ast(self):
  rel=next(iter(m.MANIFESTS));manifest=json.loads((m.P/rel).read_text());entry=manifest['source']['topk'];self.assertEqual(m.sha(entry['source']),entry['source_sha256'])
  funcs={n.name:n for n in ast.walk(ast.parse(pathlib.Path(entry['source']).read_text()))if isinstance(n,ast.FunctionDef)}
  self.assertEqual([a.arg for a in funcs['_ranks_kernel'].args.args],['output_ptr','logits_ptr','logits_stride','token_ids_ptr','vocab_size','BLOCK_SIZE'])
  self.assertEqual([a.arg for a in funcs['compute_token_logprobs'].args.args],['logits','token_ids'])
 def test_table_budget(self):
  self.assertEqual(m.IO_BUDGET['embedding_table_bytes'],248320*2560*2)
  self.assertEqual(m.IO_BUDGET['embedding_full_table_D2H_bytes'],248320*2560*2*3*2)
if __name__=='__main__':unittest.main()
