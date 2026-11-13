"""Stdlib draft receipt/actual helper contract tests, no GPU/framework imports."""
import ast,importlib.util,pathlib,sys,types,unittest
from . import modules_runtime_remaining as m
path=pathlib.Path(m.__file__).resolve()
class Tests(unittest.TestCase):
 def good(self):
  rows=[dict(family=f,case=c,value_source='contract_shaped_synthetic',descriptor=m.expected_descriptor(f,c),return_kind=m.return_kind(f),**{k:True for k in m.GATES})for f,c in m.expected_rows()]
  return dict(complete=True,records=rows,pending_transports=0,transport=dict(transfer_attempts=1,transfer_returns=1,cast_launch_attempts=1,cast_launch_returns=1,temporary_bytes=1792,same_dtype_transfer=True,source_pinned=True))
 def test_exact_order_and_count(self):
  r=self.good();self.assertTrue(m.audit(r));r['records'].reverse();self.assertFalse(m.audit(r))
 def test_each_gate_negative(self):
  for k in m.GATES:
   r=self.good();r['records'][0][k]=1;self.assertFalse(m.audit(r))
 def test_descriptor_alias_lifetime_negative(self):
  for k,v in [('descriptor',{}),('return_kind','fresh'),('value_source','actual')]:
   r=self.good();r['records'][10][k]=v;self.assertFalse(m.audit(r))
  r=self.good();r['pending_transports']=1;self.assertFalse(m.audit(r))
 def test_real_contract_errors(self):
  from tileir_vllm_runner_constructors_v1.api import validate_num_reqs
  from tileir_vllm_mamba_accepted_scatter_v1.api import validate_indices
  from tileir_vllm_runner_bool_fill_v1.api import validate_contract
  with self.assertRaises(AssertionError):validate_num_reqs(True)
  with self.assertRaises(AssertionError):validate_indices([0]*32,32)
  with self.assertRaises(AssertionError):validate_contract((32,),(1,),'torch.bool',3,34,False)
 def test_real_padding_helper_empty_no_launch_alias(self):
  from tileir_vllm_runner_bool_fill_v1.api import fill_padding_
  tensor=types.SimpleNamespace(shape=(0,),stride=lambda:(1,),dtype='torch.bool',storage_offset=lambda:3,untyped_storage=lambda:types.SimpleNamespace(nbytes=lambda:7),device=types.SimpleNamespace(type='cuda'),requires_grad=False)
  old=sys.modules.get('torch');sys.modules['torch']=types.ModuleType('torch')
  try:self.assertIs(fill_padding_(tensor,False),tensor)
  finally:
   if old is None:sys.modules.pop('torch')
   else:sys.modules['torch']=old
 def test_actual_record_helper(self):
  node=next(n for n in ast.walk(ast.parse(path.read_text()))if isinstance(n,ast.FunctionDef)and n.name=='record')
  class T:
   shape=(32,);dtype='torch.bool'
   def __init__(self,p,b):self.p=p;self.b=b
   def stride(self):return (1,)
   def numel(self):return 32
   def storage_offset(self):return 0
   def untyped_storage(self):return types.SimpleNamespace(data_ptr=lambda:self.p)
  owner=T(1,b'a');got=T(2,b'b');r=dict(records=[]);torch=types.SimpleNamespace(cuda=types.SimpleNamespace(current_stream=lambda:types.SimpleNamespace(cuda_stream=9),synchronize=lambda:None))
  env=dict(torch=torch,raw=lambda x:x.b,descriptor=m.descriptor,expected_descriptor=m.expected_descriptor,return_kind=m.return_kind,GATES=m.GATES,r=r,save=lambda:None)
  exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
  env['record']('bool_not','32',lambda:got,got,owners=(owner,));self.assertTrue(r['records'][0]['passed'])
  def bad():owner.b=b'changed';return got
  with self.assertRaises(AssertionError):env['record']('bool_not','32',bad,got,owners=(owner,))
 def test_real_position_reference_and_bounds(self):
  from tileir_vllm_pos_seq_v1.api import reference_checked
  self.assertEqual(reference_checked([1],[0,2],[5,7],[-1,-1,-9],[-2,-2]),([7,8,-9],[9,0]))
  with self.assertRaises(ValueError):reference_checked([2],[0,1],[5,7],[-1],[-2])
 def test_slot_api_public_signature(self):
  tree=ast.parse((m.P/'tileir_vllm_slot_mapping_v1/api.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='compute_slot_mappings')
  self.assertEqual([a.arg for a in fn.args.args],['tables','indices','starts','positions','sizes','enabled','out','padded'])
 def test_no_framework_imports(self):self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda.tile',sys.modules)
if __name__=='__main__':unittest.main()
