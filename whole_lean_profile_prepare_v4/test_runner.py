import ast,json,runpy,sys,unittest
from pathlib import Path
from unittest.mock import patch
from .build_runner import transformed,dump,digest
R=Path(__file__).parent
class Tests(unittest.TestCase):
 def test_frozen_source_connection(self):
  receipt=json.loads((R/'RUNNER-SOURCE.json').read_text());p=Path(receipt['original_path'])
  self.assertEqual(digest(p),receipt['original_sha256'])
  text,evidence=transformed(p.read_text());self.assertEqual(text,(R/'run.py').read_text())
  self.assertTrue(evidence['original_statements_preserved']);self.assertTrue(evidence['outer_lifecycle_ast_unchanged'])
 def test_actual_gpu_entry_pending_before_runtime_imports(self):
  with patch.object(sys,'argv',[str(R/'run.py'),'--run-gpu']),patch('whole_lean_profile_prepare_v4.admission.require_ready',side_effect=RuntimeError('Profile pending')):
   with self.assertRaisesRegex(RuntimeError,'Profile pending'):runpy.run_path(str(R/'run.py'),run_name='__main__')
  for name in ('torch','vllm','cuda.tile','whole_lean_extensions_runtime_v7.adapter'):
   self.assertNotIn(name,sys.modules)
 def test_proof_and_setup_excluded(self):
  t=ast.parse((R/'run.py').read_text());f=next(n for n in ast.walk(t)if isinstance(n,ast.FunctionDef)and n.name=='request')
  scopes=f.body[0].body;self.assertEqual(len(scopes),4)
  execution=ast.unparse(ast.Module(body=scopes[1].body,type_ignores=[]));proof=ast.unparse(ast.Module(body=scopes[2].body,type_ignores=[]));setup=ast.unparse(ast.Module(body=scopes[0].body,type_ignores=[]))
  self.assertIn('engine.add_request',execution);self.assertIn('engine.step',execution);self.assertIn('torch.cuda.synchronize()',execution)
  self.assertNotIn('adapter.finish',execution);self.assertNotIn('adapter.start',execution)
  self.assertIn('adapter.finish',proof);self.assertIn('adapter.start',setup);self.assertIsInstance(scopes[3].body[0],ast.Pass)
 def test_no_numerical_api_scopes_or_timer(self):
  t=(R/'run.py').read_text();self.assertNotIn('perf_counter',t);self.assertNotIn('public_scopes(',t)
  self.assertIn('record_shapes=False',t);self.assertIn('profile_memory=False',t)
if __name__=='__main__':unittest.main()
