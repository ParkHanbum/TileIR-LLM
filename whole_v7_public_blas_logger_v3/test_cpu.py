import ast,copy,sys,unittest
from pathlib import Path
from unittest.mock import patch
from . import gates
from .logger import summarize,Window
R=Path(__file__).parent
class Tests(unittest.TestCase):
 def test_request_ast(self):
  def req(p):return next(n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='request')
  old=req(R.parent/'whole_lean_profile_prepare_v4/run.py');new=req(R/'run.py')
  prof=next(n for n in new.body if isinstance(n,ast.With))
  body=[]
  for n in prof.body:
   if isinstance(n,ast.ImportFrom) and n.module=='whole_v7_public_blas_logger_v3.logger':continue
   if isinstance(n,ast.With) and ast.unparse(n.items[0].context_expr)=='Window(a.output, mode)':body+=n.body;continue
   if isinstance(n,ast.Assign) and ast.unparse(n.value)=='_log.verify()':continue
   if isinstance(n,ast.Expr) and ast.unparse(n)=='save()':continue
   body.append(n)
  prof.body=body
  self.assertEqual(ast.dump(old),ast.dump(new))
 def test_roots(self):
  with patch.object(gates,'require_ready',return_value={'passed':True}):
   for i in range(3):self.assertTrue(gates.qualification(R.parent/(R.name+'_process'+str(i)))['passed'])
   for p in (R,R/'process0',R.parent/'wrong',R.parent/(R.name+'_process3')):
    with self.assertRaises(AssertionError):gates.qualification(p)
 def test_parser(self):self.assertEqual(summarize('I! cuBLAS (v13) function cublasStatus_t cublasGemmEx(x) called:\n'),{'cublasGemmEx':1})
 def test_no_framework(self):self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda.tile',sys.modules)
 def test_ast(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
 def test_launcher_destination(self):
  s=(R/'launch.py').read_text();self.assertIn("R=SOURCE_ROOT.parent/(SOURCE_ROOT.name+'_process'+str(args.process_index))",s);self.assertIn("env['PUBLIC_BLAS_AUDIT_ROOT']=str(R)",s);self.assertIn('audit(result,root=R)',s);self.assertIn('preflight(R)',s);self.assertIn("str(SOURCE_ROOT/'run.py')",s)
