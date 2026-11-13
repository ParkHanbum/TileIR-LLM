import ast,copy,importlib.util,pathlib,sys,types,unittest
from . import modules_pointwise_remaining as m
p=pathlib.Path(m.__file__).resolve()
class Tests(unittest.TestCase):
 def good(self):
  return dict(complete=True,records=[dict(identity=i,readonly=True,alias=True,current_stream=True,return_contract=True,repeat_equal=True,outputs=[dict(descriptor=m.expected(s),raw_equal=True,raw_mismatches=0,candidate_nonfinite=0,reference_nonfinite=0) for s in m.shapes(*i[:2])]) for i in m.identities()])
 def test_identity(self):
  self.assertTrue(m.audit(self.good()))
  for mode in ('duplicate','missing','reverse'):
   r=self.good()
   if mode=='duplicate':r['records'][1]=copy.deepcopy(r['records'][0])
   elif mode=='missing':r['records'].pop()
   else:r['records'].reverse()
   self.assertFalse(m.audit(r))
 def test_metrics(self):
  for key,value in [('raw_equal',False),('raw_mismatches',1),('raw_mismatches',False),('candidate_nonfinite',1),('reference_nonfinite',1)]:
   r=self.good();r['records'][0]['outputs'][0][key]=value;self.assertFalse(m.audit(r))
 def test_descriptors(self):
  for k,v in [('shape',[32,4096]),('elements',1),('dtype','torch.float32'),('stride',[1,1]),('offset',1)]:
   r=self.good();r['records'][0]['outputs'][0]['descriptor'][k]=v;self.assertFalse(m.audit(r))
 def test_contracts(self):
  for k in ('readonly','alias','current_stream','return_contract','repeat_equal'):
   r=self.good();r['records'][3][k]=False;self.assertFalse(m.audit(r))
 def test_real_byte_comparator_corruption(self):
  a=bytes(range(32));self.assertTrue(m.compare_bytes(a,a)['raw_equal'])
  for at in (0,15,31):
   b=bytearray(a);b[at]^=1;got=m.compare_bytes(a,bytes(b));self.assertFalse(got['raw_equal']);self.assertEqual(got['raw_mismatches'],1)
  with self.assertRaises(AssertionError):m.compare_bytes(a,a[:-1])
 def test_actual_load_helper(self):
  tree=ast.parse(p.read_text());fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='load');calls=[];value=object()
  env={'torch':types.SimpleNamespace(load=lambda *a,**kw:(calls.append((a,kw)) or value))}
  exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),env)
  self.assertIs(env['load']({'path':'saved.pt'}),value);self.assertEqual(calls,[(('saved.pt',),dict(map_location='cpu',weights_only=True))])
 def test_actual_banks_and_active_api(self):
  q,s=m.banks();self.assertEqual(len(q),2);self.assertEqual(len(s),2)
  source=(m.P/'lean_public_composition_v1/model.py').read_text()
  self.assertIn('from tileir_vllm_qk_rope_v1 import fused_qk_rmsnorm_rope_gate',source)
  self.assertIn('from tileir_vllm_silu_mul_v2.api import silu_and_mul',source)
  for sample in q:
   for entry,shape in zip(sample['outputs'],m.shapes('qk_rope',sample['phase'])):self.assertEqual(entry['metadata']['shape'],shape)
 def test_no_framework_or_main_extraction(self):
  self.assertNotIn('torch',sys.modules)
  src=p.read_text();self.assertNotIn('ast.walk',src);self.assertNotIn('sys.argv',src)
if __name__=='__main__':unittest.main()
