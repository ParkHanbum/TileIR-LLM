import unittest,json,copy,ast
from pathlib import Path
from tileir_vllm_sampler_fp32_copy_v1.api import validate_metadata
from sampler_fp32_copy_qualification_v1.run import check
from sampler_fp32_copy_qualification_v1.launch import result_gate
class Tests(unittest.TestCase):
 def test_metadata(self):
  m=dict(shape=(32,248320),stride=(248320,1),offset=0,dtype='torch.bfloat16');self.assertEqual(validate_metadata(m),m['shape'])
  for k,v in [('dtype','torch.float32'),('offset',1),('shape',(32,2560)),('stride',(248321,1))]:
   b=dict(m);b[k]=v
   with self.assertRaises(ValueError):validate_metadata(b)
 def test_manifest(self):
  p=Path(__file__).resolve().parent.parent/'sampler_fp32_copy_runtime_capture_v1/samplerFP32CopyCapture01/bank/MANIFEST.json';m=json.loads(p.read_text());self.assertTrue(check(m));m['samples'][0]['immediate_output']['metadata']['dtype']='torch.bfloat16'
  with self.assertRaises(AssertionError):check(m)
 def test_gate(self):self.assertFalse(result_gate({})['raw_pass'])
 def test_source_syntax_cache_order(self):
  R=Path(__file__).parent
  for f in R.glob('*.py'):ast.parse(f.read_text())
  s=(R/'run.py').read_text();self.assertLess(s.index("('CUDA_TILE_CACHE_DIR','tile')"),s.index('import torch'))
if __name__=='__main__':unittest.main()
