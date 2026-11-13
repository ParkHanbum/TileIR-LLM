"""Draft stdlib-only contract/audit tests. GPU execution remains pending."""
import copy
import importlib.util
from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest.mock import patch

# Works both as an external review draft and after parent integration.
try:
 from . import modules_norm_conv as m
except ImportError:
 spec=importlib.util.spec_from_file_location('norm_conv_draft',Path(__file__).with_name('modules_norm_conv.py'))
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture():
 rows=[]
 for name,backend in m.EXPECTED:
  counts=[4341760]if name=='norm_no_residual' else [4341760,4341760]if name=='norm_with_residual' else [262144,786432]
  metrics=[dict(elements=n,raw_equal=True,strict_pass=True,raw_mismatches=0,strict_failures=0,candidate_nonfinite=0,reference_nonfinite=0)for n in counts]
  rows.append(dict(case=name,backend=backend,metrics=metrics,contract=True,input_contract=True,same_stream=True,corruption_detected=True,fresh_outputs=True,full_input_backing_readonly=True,selected_state_exact=True,state_minspan_exact=True,x_minspan_exact=True,weight_indices_readonly=True,distinct_same_storage_return=True))
 return dict(complete=True,context_restored=True,timing=False,scope='two prefill norm boundaries and one decode convolution; not complete families',records=rows)

class Tests(unittest.TestCase):
 def test_create_only_partial_receipt(self):
  with tempfile.TemporaryDirectory() as root:
   out=m.prepare_output(Path(root)/'fresh')
   partial=dict(complete=False,records=[dict(case='completed')],error='injected failure')
   m.save_partial(out,partial)
   self.assertEqual(json.loads((out/'RESULT.json').read_text()),partial)
   with self.assertRaises(FileExistsError):m.prepare_output(out)
   with self.assertRaises(FileExistsError):m.save_partial(out,{})
 def test_good_gate(self):self.assertTrue(m.audit(fixture())['passed'])
 def test_missing_duplicate_and_order(self):
  for mode in ('missing','duplicate','order'):
   r=fixture()
   if mode=='missing':r['records'].pop()
   elif mode=='duplicate':r['records'][1]=r['records'][0]
   else:r['records'].reverse()
   self.assertFalse(m.audit(r)['passed'])
 def test_false_contracts(self):
  for field in ('same_stream','input_contract','corruption_detected','fresh_outputs','full_input_backing_readonly'):
   r=fixture();r['records'][0][field]=False;self.assertFalse(m.audit(r)['passed'],field)
  for field in ('selected_state_exact','state_minspan_exact','x_minspan_exact','weight_indices_readonly','distinct_same_storage_return'):
   r=fixture();r['records'][-1][field]=False;self.assertFalse(m.audit(r)['passed'],field)
 def test_numerical_failures(self):
  for field,val in [('raw_equal',False),('strict_pass',False),('raw_mismatches',1),('strict_failures',1),('candidate_nonfinite',1),('reference_nonfinite',1),('elements',1)]:
   r=fixture();r['records'][0]['metrics'][0][field]=val;self.assertFalse(m.audit(r)['passed'])
 def test_context_and_missing(self):
  r=fixture();r['context_restored']=False;self.assertFalse(m.audit(r)['passed']);self.assertFalse(m.audit({})['passed'])
 def test_corruption_detector(self):
  import torch
  value=torch.tensor([1.,-2.,0.,3.],dtype=torch.bfloat16);original=value.view(torch.int16).clone()
  self.assertTrue(m.assert_corruption_detected(value,torch))
  self.assertTrue(torch.equal(value.view(torch.int16),original))
  with self.assertRaises(AssertionError):m.assert_corruption_detected(torch.empty(0,dtype=torch.bfloat16),torch)
 def test_always_pass_comparator_is_detected(self):
  import torch
  from norm_runtime_qualification_v2.compare import metrics
  value=torch.tensor([1.,2.],dtype=torch.bfloat16)
  good=metrics(value,value)
  with self.assertRaisesRegex(AssertionError,'accepted deliberate'):
   m.assert_corruption_detected(value,torch,compare=lambda a,b:good)
 def test_always_reject_comparator_is_detected(self):
  import torch
  with self.assertRaises(AssertionError):
   m.assert_corruption_detected(torch.ones(2,dtype=torch.bfloat16),torch,compare=lambda a,b:dict(raw_equal=False))
 def test_minspan_not_original_fullowner(self):
  _,c=m.manifests();md=c['state_descriptor'];self.assertLess(m.span(md)*2,md['storage_nbytes']);self.assertLessEqual(m.span(md)*2,700*1024*1024)
 def test_manifest_and_scope(self):
  norms,c=m.manifests();self.assertEqual(len(norms),2);self.assertEqual(c['phase'],'decode');self.assertEqual(len(c['slots']),32)
  self.assertFalse(any('tileY' in path for path in m.dependencies()))
 def test_source_pins(self):
  for path,h in m.SOURCE_PINS.items():self.assertEqual(m.sha(path),h)
 def test_bad_span(self):
  for md in [dict(shape=[2],stride=[-1],storage_offset=0),dict(shape=[2],stride=[1],storage_offset=-1)]:
   with self.assertRaises(AssertionError):m.span(md)
 def test_original_epsilon_error_without_backend(self):
  from tileir_vllm_gemma_norm_v3.api import validate
  for eps in (True,0,-1,float('nan'),float('inf')):
   with patch.dict(sys.modules,{'torch':object()}):
    with self.assertRaises(ValueError):validate(None,None,eps)
 def test_original_conv_duplicate_and_bounds(self):
  from tileir_indexed_decode_conv_v1.api import validate_metadata
  x=dict(shape=(32,8192),stride=(8192,1),dtype='torch.bfloat16',offset=0)
  state=dict(shape=(311,8192,3),stride=(1081344,1,8192),dtype='torch.bfloat16',offset=0)
  w=dict(shape=(8192,4),stride=(4,1),dtype='torch.bfloat16',offset=0)
  idx=dict(shape=(32,),stride=(1,),dtype='torch.int32',offset=0)
  self.assertEqual(validate_metadata(x,state,w,idx,list(range(1,33))),32)
  for slots in ([1]*32,[400]*32):
   with self.assertRaises(ValueError):validate_metadata(x,state,w,idx,slots)

if __name__=='__main__':unittest.main()
