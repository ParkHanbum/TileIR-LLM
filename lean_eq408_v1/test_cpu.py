import unittest,types,sys
from pathlib import Path
from lean_eq408_v1.api import Checks,LIMITS
from lean_eq408_v1.bridge import GDNBodyBinding,combined_attention
from lean_public_composition_v2.attention_runner import AttentionCore
class C:
 active=True;failed=None
 def latch(self,k,e):self.failed=k
class X:
 dtype='torch.float32'
 def __init__(self,gdn=False):self.shape=(32,32,128,128)if gdn else(32,4);self.gdn=gdn
 def stride(self):return(524288,16384,128,1)if self.gdn else(0,0)
 def storage_offset(self):return 0
class Tests(unittest.TestCase):
 def test_full_census(self):
  c=C();q=Checks(c,lambda x,v:True);q.start()
  for site,n in LIMITS.items():
   for _ in range(n):self.assertIs(q.predicate(site,X(site=='gdn.initial'),0 if site=='gdn.initial'else 1),True)
  self.assertEqual(q.finish()['calls'],408)
 def test_false_preserved(self):
  q=Checks(C(),lambda x,v:False);q.start();self.assertIs(q.equal_one('q_descale',X()),False)
 def test_cap_before_candidate(self):
  calls=[];q=Checks(C(),lambda x,v:(calls.append(1)or True));q.start()
  for _ in range(128):q.equal_one('q_descale',X())
  with self.assertRaises(AssertionError):q.equal_one('q_descale',X())
  self.assertEqual(len(calls),128)
 def test_latch(self):
  calls=[]
  def fail(*a):calls.append(1);raise ValueError('test')
  q=Checks(C(),fail);q.start()
  with self.assertRaises(ValueError):q.equal_one('q_descale',X())
  with self.assertRaises(AssertionError):q.equal_one('k_descale',X())
  self.assertEqual(calls,[1])
 def test_gdn_body_restore(self):
  path=Path('/home/n/work/TileIR/public-tileir-blas-20261111/lean26_gdn_finite_runtime_v1/apply.candidate.py.txt');ns={'_gdn_finite':object()};exec(compile(path.read_text().strip(),str(path),'exec'),ns)
  part=types.SimpleNamespace(body=ns['apply']);orig=part.body;q=Checks(C(),lambda *a:True);b=GDNBodyBinding(part,q);b.install()
  self.assertIs(part.body.__globals__['_gdn_finite'],ns['_gdn_finite']);self.assertIsNot(part.body,orig);b.close();self.assertIs(part.body,orig)
 def test_attention_dual_namespace(self):
  finite=types.SimpleNamespace(finite=lambda *a:True);q=Checks(C(),lambda *a:True);body=combined_attention(AttentionCore.apply,finite,q)
  self.assertIs(body.__globals__['_finite'],finite.finite);self.assertEqual(body.__globals__['_equal_one'],q.equal_one)
 def test_no_diagnostics(self):
  s=Path(__file__).with_name('api.py').read_text()
  for token in ['.cpu(','.clone(','.numpy(','.set_(','.fsync(']:self.assertNotIn(token,s)
  self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda',sys.modules)
if __name__=='__main__':unittest.main()
