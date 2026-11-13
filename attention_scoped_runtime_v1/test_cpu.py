import unittest,tempfile,json,copy,ast
from pathlib import Path
from .adapter import classify,census,MODULES,Adapter,validate_scalars
from .preflight import verify,sha
class Tests(unittest.TestCase):
 def test_census(self):
  rows=[dict(module=m,length=L,supported=L<=64,passed=True) for m in MODULES for L in range(53,69)]
  self.assertTrue(census(rows));self.assertEqual(sum(r['supported'] for r in rows),96)
  for bad in (rows[:-1],rows+[rows[0]], [dict(r,passed=False) for r in rows]):
   with self.assertRaises(AssertionError):census(bad)
 def test_lengths(self):
  self.assertEqual(classify([53*i for i in range(33)],[53]*32),(53,True))
  for L in range(54,69):self.assertEqual(classify(list(range(33)),[L]*32),(L,L<=64))
  for L in (52,69):
   with self.assertRaises(AssertionError):classify(list(range(33)),[L]*32)
 def test_transition(self):
  with tempfile.TemporaryDirectory() as t:
   a=Adapter(Path(t)/'out')
   with self.assertRaises(AssertionError):a.begin('replace')
   a.begin('shadow');a.failed=True
   with self.assertRaises(AssertionError):a.begin('replace')
 def test_late_scope(self):
  src=Path(__file__).with_name('adapter.py').read_text()
  self.assertIn("self.mode=='shadow' and ctx['name'].endswith('layers.3.self_attn.attn') and ctx['length'] in (65,68)",src)
  self.assertIn("target=torch.empty_like(out) if self.mode=='shadow' else out",src)
  self.assertIn("assert self.digest(out)==native_hash",src)
 def test_cleanup_wiring(self):
  tree=ast.parse(Path(__file__).with_name('run.py').read_text());tries=[n for n in ast.walk(tree) if isinstance(n,ast.Try) and n.finalbody]
  self.assertTrue(any(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='final_cleanup' for x in t.finalbody for n in ast.walk(x)) for t in tries))
 def test_cache_preflight_missing(self):
  root=Path(__file__).parent
  self.assertTrue(verify(root,False))
  with self.assertRaises(FileNotFoundError):verify(root,True)
 def test_fixture_scalars(self):
  root=Path(__file__).parent.parent
  r=json.loads((root/'attention_core_runtime_capture_v1/attentionCoreCapture01/bank/MANIFEST.json').read_text())
  for c in r['calls']:validate_scalars(c['scalar_args'])
  bad=copy.deepcopy(r['calls'][0]['scalar_args']);bad['fa_version']=4
  with self.assertRaises(AssertionError):validate_scalars(bad)
 def test_callgate_negative(self):
  from .result_gate import audit
  self.assertFalse(audit({})['passed'])
if __name__=='__main__':unittest.main()
