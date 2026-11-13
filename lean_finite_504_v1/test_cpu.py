import unittest,types,sys,ast
from pathlib import Path
from unittest.mock import patch
from . import api,bridge
class T(unittest.TestCase):
 def manager(self):
  c=types.SimpleNamespace(active=True,failed=None)
  def latch(k,e):c.failed=repr(e)
  c.latch=latch
  with patch.object(api,'verify',return_value={'passed':True}):s=api.Checks(c)
  return c,s
 def test504(self):
  c,s=self.manager();s.start();x=types.SimpleNamespace(untyped_storage=lambda:types.SimpleNamespace(nbytes=lambda:64))
  with patch.dict(sys.modules,{'tileir_validation_primitives_v3.api':types.SimpleNamespace(all_finite=lambda x:True),'tileir_kv_strided_finite_v1.api':types.SimpleNamespace(all_finite_value=lambda x:True)}):
   for k,n in api.LIMITS.items():
    for _ in range(n):self.assertIs(s.finite(k,x),True)
  self.assertEqual(s.finish()['calls'],504)
 def test_false_and_failure(self):
  c,s=self.manager();s.start();x=types.SimpleNamespace(untyped_storage=lambda:types.SimpleNamespace(nbytes=lambda:64))
  with patch.dict(sys.modules,{'tileir_validation_primitives_v3.api':types.SimpleNamespace(all_finite=lambda x:False),'tileir_kv_strided_finite_v1.api':types.SimpleNamespace(all_finite_value=lambda x:False)}):
   self.assertIs(s.finite('kv.key',x),False);self.assertIsNone(c.failed)
   s.counts['kv.key']=128
   with self.assertRaises(AssertionError):s.finite('kv.key',x)
   with self.assertRaises(AssertionError):s.finite('post.a',x)
 def test_single_body(self):
  from lean_public_composition_v2.attention_runner import AttentionCore
  a=lambda site,x:True;b=lambda name,x:True
  f=bridge.attention_body(AttentionCore.apply,a,b)
  self.assertIs(f.__globals__['_finite'],a);self.assertIs(f.__globals__['_equal_one'],b)
  source=(bridge.P/'floating_guard_504_design_v1/attention.original.py.txt').read_text()
  new=source.replace('bool(torch.isfinite(out).all())',"_finite('attention.out',out)").replace('bool((t==1).all())','_equal_one(n,t)')
  ast.parse(new);self.assertIn('result is out and',new)
 def test_no_oracle(self):
  tree=ast.parse(Path(api.__file__).read_text());text=ast.unparse(tree)
  self.assertNotIn('.cpu(',text);self.assertNotIn('torch.isfinite',text);self.assertNotIn('torch.equal',text)
  self.assertFalse(any(k in sys.modules for k in ('torch','vllm','cuda.tile')))
if __name__=='__main__':unittest.main()
