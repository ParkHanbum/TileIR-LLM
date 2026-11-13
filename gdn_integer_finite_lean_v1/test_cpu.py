import unittest
from pathlib import Path
from types import SimpleNamespace
from .core import PrefillCore,verify_finite_binding,replace_specs
from gdn_integer_finite_typed_v1.core import PrefillCore as Typed
from lean26_gdn_finite_runtime_v1.core import PrefillCore as Original
from lean_eq408_v1.bridge import GDNBodyBinding
class Tests(unittest.TestCase):
 def part(self):
  part=object.__new__(PrefillCore)
  path=Path(__file__).parent.parent/'lean26_gdn_finite_runtime_v1/apply.candidate.py.txt'
  ns={'_gdn_finite':part.finite};exec(compile(path.read_text(),str(path),'exec'),ns);part.body=ns['apply'];return part
 def test_actual_eq_body_keeps_bound_method(self):
  p=self.part();old=p.body;self.assertTrue(verify_finite_binding(p));b=GDNBodyBinding(p,SimpleNamespace(initial_zero=lambda x:True));b.install()
  self.assertIsNot(p.body,old);self.assertTrue(verify_finite_binding(p));self.assertIs(p.body.__globals__['_gdn_finite'].__func__,Typed.finite)
  b.close();self.assertIs(p.body,old);self.assertTrue(verify_finite_binding(p))
 def test_wrong_bound_owner_rejects(self):
  p=self.part();other=self.part();p.body.__globals__['_gdn_finite']=other.finite
  with self.assertRaises(RuntimeError):verify_finite_binding(p)
 def test_original_finite_rejects(self):
  p=self.part();p.body.__globals__['_gdn_finite']=Original.finite.__get__(p)
  with self.assertRaises(RuntimeError):verify_finite_binding(p)
 def test_specs_exact_only(self):
  sentinel=object();s=(('other',sentinel,1),('gdn_prefill_core',Original,24));r=replace_specs(s)
  self.assertEqual(r[0],s[0]);self.assertIs(r[1][1],PrefillCore)
  with self.assertRaises(ValueError):replace_specs(r)
 def test_eq_restore_foreign_rejected(self):
  p=self.part();b=GDNBodyBinding(p,SimpleNamespace(initial_zero=lambda x:True));b.install();p.body=lambda:None
  with self.assertRaises(AssertionError):b.close()
