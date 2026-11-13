import unittest,copy,json
from pathlib import Path
from tileir_vllm_hidden_gather_v1.api import validate_metadata
from runner_hidden_gather_qualification_v1.run import check
from runner_hidden_gather_qualification_v1.launch import result_gate
class Tests(unittest.TestCase):
 def setUp(self):
  self.i=dict(shape=(32,),stride=(1,),dtype='torch.int64',offset=0);self.x=dict(shape=(1696,2560),stride=(2560,1),dtype='torch.bfloat16',offset=0)
 def test_repeated_reordered(self):self.assertEqual(validate_metadata(self.i,self.x,list(reversed(range(32)))) ,(32,2560));validate_metadata(self.i,self.x,[7]*32)
 def test_bounds(self):
  for v in (-1,1696):
   with self.assertRaises(ValueError):validate_metadata(self.i,self.x,[v]*32)
 def test_contract(self):
  for key,value in [('dtype','torch.float32'),('stride',(2561,1)),('shape',(248320,2560)),('offset',1)]:
   d=dict(self.x);d[key]=value
   with self.assertRaises(ValueError):validate_metadata(self.i,d,list(range(32)))
 def test_manifest(self):
  p=Path(__file__).resolve().parent.parent/'runner_hidden_gather_runtime_capture_v1/hiddenGatherCapture01/bank/MANIFEST.json';m=json.loads(p.read_text());self.assertTrue(check(m));m['samples'][0]['output']['metadata']['storage_offset']=1
  with self.assertRaises(AssertionError):check(m)
 def test_gate_failclosed(self):self.assertFalse(result_gate({})['raw_pass'])
if __name__=='__main__':unittest.main()
