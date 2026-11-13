import unittest,sys,ast
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from gdn_post_conv_qualification_v2.run import result_gate
class Tests(unittest.TestCase):
 def test_success(self):self.assertTrue(result_gate({'complete':True,'gpu_executed':True,'records':[{'passed':True}]*4})['raw_pass'])
 def test_partial_not_success(self):self.assertFalse(result_gate({'complete':True,'gpu_executed':True,'records':[{'passed':True}]*3})['raw_pass'])
 def test_raw_hold_distinct(self):self.assertTrue(result_gate({'complete':False,'raw_difference_hold':True,'records':[{'passed':False}]})['raw_difference_hold'])
 def test_no_gpu_import(self):self.assertNotIn('torch',sys.modules);self.assertNotIn('vllm',sys.modules)
 def test_all_sources_ast(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
if __name__=='__main__':unittest.main()
