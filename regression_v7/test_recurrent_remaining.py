"""CPU-only draft tests; numeric GPU replay is pending independent review."""
import copy,importlib.util,json,sys,unittest
from pathlib import Path
try:
 from . import modules_recurrent_remaining as m
except ImportError:
 spec=importlib.util.spec_from_file_location('remaining_recurrent_draft',Path(__file__).with_name('modules_recurrent_remaining.py'))
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def good():return dict(raw_equal=True,raw_different_elements=0,strict_failures=0,candidate_nonfinite=0,reference_nonfinite=0,strict_atol=1e-5,strict_rtol=1e-5)
def fixture():
 specs=m.selected(m.manifests());rows=[]
 for i,name in enumerate(m.CASES):
  s=specs[i][0]if i<4 else specs[i]
  outs=s['outputs']if i<4 else [s['output']]if i==4 else [s['outputs'][k]for k in ('q','k','v','g','beta')]
  compared=list(outs)+([s['selected_state_after']]if i==4 else [])
  comparisons=[]
  for item in compared:
   md=m.captured_comparison_descriptor(item);comparisons.append(dict(good(),reference_descriptor=md.copy(),actual_descriptor=md.copy(),elements=md['elements'],logical_bytes=md['logical_bytes']))
  for backend in ('native','v7'):
   rows.append(dict(case=name,backend=backend,comparisons=copy.deepcopy(comparisons),descriptors=[dict(t['metadata'])for t in outs],readonly=True,descriptor_exact=True,fresh_outputs=True,same_stream=True,negative_detected=True,state_metadata_contract=True))
 return dict(complete=True,context_restored=True,timing=False,records=rows)

class Tests(unittest.TestCase):
 def test_valid(self):self.assertTrue(m.audit(fixture())['passed'])
 def test_missing_duplicate_order(self):
  for mode in range(3):
   r=fixture()
   if mode==0:r['records'].pop()
   elif mode==1:r['records'][1]=r['records'][0]
   else:r['records'].reverse()
   self.assertFalse(m.audit(r)['passed'])
 def test_contract_failures(self):
  for name in ('readonly','descriptor_exact','fresh_outputs','same_stream','negative_detected','state_metadata_contract'):
   r=fixture();r['records'][0][name]=False;self.assertFalse(m.audit(r)['passed'])
 def test_raw_strict_finite_failures(self):
  for name,val in [('raw_equal',False),('raw_different_elements',1),('strict_failures',1),('candidate_nonfinite',1),('reference_nonfinite',1),('strict_atol',1.)]:
   r=fixture();r['records'][-1]['comparisons'][0][name]=val;self.assertFalse(m.audit(r)['passed'])
 def test_output_descriptor_mutation(self):
  for key,value in [('shape',[1]),('stride',[1]),('dtype','torch.float64'),('storage_offset',7)]:
   r=fixture();r['records'][0]['descriptors'][0][key]=value;self.assertFalse(m.audit(r)['passed'])
 def test_comparison_descriptor_tampering(self):
  for index in (0,8):
   for field,value in [('shape',[1]),('dtype','torch.float64'),('elements',1),('logical_bytes',1)]:
    r=fixture();c=r['records'][index]['comparisons'][-1];c['actual_descriptor'][field]=value
    self.assertFalse(m.audit(r)['passed'],(index,field))
   for field,value in [('elements',1),('elements',True),('logical_bytes',1),('raw_different_elements',False)]:
    r=fixture();r['records'][index]['comparisons'][-1][field]=value;self.assertFalse(m.audit(r)['passed'])
 def test_actual_cpu_comparison_descriptors(self):
  import torch
  from gdn_post_conv_qualification_v2.run import compare
  for dtype in (torch.bfloat16,torch.float32):
   value=torch.ones((2,3),dtype=dtype);r=m.actual_comparison(value,value.clone(),torch,compare)
   m.require_metric(r);self.assertEqual(r['elements'],6);self.assertEqual(r['logical_bytes'],6*value.element_size())
   self.assertEqual(r['actual_descriptor'],r['reference_descriptor'])
   with self.assertRaises(AssertionError):m.actual_comparison(value,value.reshape(3,2),torch,compare)
   with self.assertRaises(AssertionError):m.actual_comparison(value,value.double(),torch,compare)
 def test_missing_comparison(self):
  r=fixture();r['records'][-1]['comparisons'].pop();self.assertFalse(m.audit(r)['passed'])
 def test_cleanup_failure(self):
  r=fixture();r['cleanup_error']='sync';self.assertFalse(m.audit(r)['passed'])
 def test_real_comparator_negatives(self):
  import torch
  from gdn_post_conv_qualification_v2.run import compare
  for dtype in (torch.bfloat16,torch.float32):
   value=torch.tensor([1.,-2.,0.,3.],dtype=dtype);before=value.view(torch.uint8).clone()
   self.assertTrue(m.negative(value,torch,compare));self.assertTrue(torch.equal(before,value.view(torch.uint8)))
 def test_broken_comparator(self):
  import torch
  value=torch.ones(2,dtype=torch.float32)
  with self.assertRaises(AssertionError):m.negative(value,torch,lambda *a:good())
  with self.assertRaises(AssertionError):m.negative(value,torch,lambda *a:dict(raw_equal=False))
 def test_manifest_cases(self):
  specs=m.selected(m.manifests());self.assertEqual(len(specs),6)
  self.assertEqual([x[0]['phase']for x in specs[:2]],['prefill','decode'])
  self.assertEqual([x[0]['kind']for x in specs[2:4]],['no_residual','with_residual'])
 def test_original_gated_epsilon_contract(self):
  from tileir_vllm_gated_norm_ordered_v1.api import validate_metadata
  x=dict(shape=(32,32,128),stride=(4096,128,1),dtype='torch.bfloat16',offset=0);w=dict(shape=(128,),stride=(1,),dtype='torch.bfloat16',offset=0)
  self.assertEqual(validate_metadata(x,x,w,1e-6),(32,1e-6))
  for epsilon in (True,0.,-1.,float('nan')):
   with self.assertRaises(ValueError):validate_metadata(x,x,w,epsilon)
 def test_original_gated_bad_stride(self):
  from tileir_vllm_gated_norm_ordered_v1.api import validate_metadata
  x=dict(shape=(32,32,128),stride=(4096,128,2),dtype='torch.bfloat16',offset=0);w=dict(shape=(128,),stride=(1,),dtype='torch.bfloat16',offset=0)
  with self.assertRaises(ValueError):validate_metadata(x,x,w,1e-6)
 def test_source_provenance(self):
  for path,h in m.SOURCE_PINS.items():self.assertEqual(m.sha(path),h)
 def test_imported_cpu_names(self):
  self.assertEqual(m.cpu_tests()[-1],'regression_v7.test_recurrent_remaining')
  for module in m.cpu_tests()[:-1]:self.assertTrue((m.P/(module.replace('.','/')+'.py')).exists())

if __name__=='__main__':unittest.main()
