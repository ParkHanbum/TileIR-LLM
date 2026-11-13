import unittest,tempfile,sys,ast,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from gdn_post_conv_runtime_integration_v2.adapter import Adapter
from gdn_post_conv_runtime_integration_v2.result_gate import audit
class Tests(unittest.TestCase):
 def test_replace_requires_complete_shadow(self):
  with tempfile.TemporaryDirectory() as t:
   a=Adapter(Path(t)/'out')
   with self.assertRaises(AssertionError):a.start('replace')
 def test_exact24_census(self):
  with tempfile.TemporaryDirectory() as t:
   a=Adapter(Path(t)/'out');a.live={str(i):None for i in range(24)};a.rows=[{'path':str(i),'passed':True} for i in range(24)];a.native_gate_calls=a.tileir_gate_calls=24
   self.assertTrue(a.finish()['shadow_pass']);a.start('replace');self.assertEqual(a.rows,[])
 def test_missing_duplicate_failure_blocks(self):
  for mode in ('missing','duplicate','failed'):
   with tempfile.TemporaryDirectory() as t:
    a=Adapter(Path(t)/'out');a.live={str(i):None for i in range(24)};a.rows=[{'path':str(i),'passed':True} for i in range(24)];a.native_gate_calls=a.tileir_gate_calls=24
    if mode=='missing':a.rows.pop()
    elif mode=='duplicate':a.rows[-1]['path']='0'
    else:a.rows[-1]['passed']=False
    with self.assertRaises(AssertionError):a.finish()
    self.assertFalse(a.shadow_pass)
 def test_no_backend_import(self):self.assertNotIn('torch',sys.modules);self.assertNotIn('vllm',sys.modules)
 def test_numerical_source_unchanged(self):
  p=Path(__file__).resolve().parent.parent
  pins=json.loads((p/'tileir_vllm_post_conv_v2/CPU-PINS.json').read_text());import hashlib
  for name in ('api.py','kernels.py'):
   f=p/'tileir_vllm_post_conv_v2'/name;self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),pins[str(f)])
 def test_actual_source_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
