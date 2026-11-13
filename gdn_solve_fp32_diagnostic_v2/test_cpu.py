import ast,unittest,json
from pathlib import Path
from .run import result_gate
R=Path(__file__).parent;P=R.parent
S=Path("/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/vllm/third_party/flash_linear_attention/ops/solve_tril.py")
class Tests(unittest.TestCase):
 def test_ast(self):
  for p in R.glob('*.py'):ast.parse(p.read_text())
 def test_native_arithmetic_unchanged(self):
  original=next(n for n in ast.parse(S.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='merge_16x16_to_64x64_inverse_kernel')
  probe=next(n for n in ast.parse((R/'native_probe.py').read_text()).body if isinstance(n,ast.FunctionDef))
  self.assertEqual(ast.dump(ast.Module(body=original.body,type_ignores=[])),ast.dump(ast.Module(body=probe.body[:-1],type_ignores=[])))
  self.assertEqual(len(probe.body[-1].body),10)
 def test_own_arithmetic_unchanged(self):
  a=next(n for n in ast.parse((P/'tileir_gdn_solve_tril_v5/kernels.py').read_text()).body if isinstance(n,ast.FunctionDef))
  b=next(n for n in ast.parse((R/'own_probe.py').read_text()).body if isinstance(n,ast.FunctionDef))
  self.assertEqual(ast.dump(ast.Module(body=a.body,type_ignores=[])),ast.dump(ast.Module(body=b.body[:-1],type_ignores=[])))
  self.assertEqual(len(b.body[-1].body),10)
 def test_banks_and_size(self):
  m=json.loads((R/'MANIFEST.json').read_text());self.assertEqual(set(m['banks']),{'layer0','layer12'});self.assertEqual(10*32*16*16*4,327680)
 def test_binding_gate(self):
  self.assertFalse(result_gate({})['raw_pass']);self.assertFalse(result_gate({'complete':True,'banks':[{},{}]})['raw_pass'])
 def test_runtime_order(self):
  s=(R/'run.py').read_text();self.assertLess(s.index('out=solver.solve_tril'),s.index("for arm in ('native_probe','own_probe')"));self.assertIn('config=auto.best_config',s);self.assertIn("if not(valid(metric) and preserved)",s)

 def test_live_winner_before_comparison(self):
  s=(R/'run.py').read_text();self.assertLess(s.index('config=auto.best_config'),s.index("check(out,refs['native'],f'native{rep}')"));self.assertIn("winner!=pre['historical_config']",s);self.assertIn('mismatch_sha=sha(failure)',s)
 def test_preseed_receipt(self):
  p=json.loads((R/'PRESEED.json').read_text());self.assertEqual(p['count'],len(p['files']));self.assertEqual(p['historical_config']['num_warps'],4);self.assertEqual(p['historical_config']['num_stages'],3)
