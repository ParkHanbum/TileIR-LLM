import unittest,ast,json,sys,types
from pathlib import Path
from eq408_runtime_v1.adapter import LIMITS,Checks,Attention,GDN,SPECS
from eq408_runtime_v1.gates import eq_gate
from eq408_runtime_v1.finite504_bridge import attention_method
from lean_public_composition_v2.attention_runner import AttentionCore
class Tests(unittest.TestCase):
 def test_census(self):
  self.assertEqual(sum(LIMITS.values()),408);self.assertEqual(sum(v for k,v in LIMITS.items()if k.startswith('attention.')),384)
 def test_only_two_components(self):
  from lean26_gdn_finite_runtime_v1.adapter import SPECS as old
  changed=[a[0]for a,b in zip(SPECS,old)if a!=b];self.assertEqual(changed,['gdn_prefill_core','attention_core'])
 def test_explicit_finite_bridge(self):
  f=lambda *a:True;e=lambda *a:True
  body=attention_method(AttentionCore.apply,f,e)
  self.assertIs(body.__globals__['_finite'],f);self.assertIs(body.__globals__['_eq_all'],e)
  self.assertNotIn('_eq_all',AttentionCore.apply.__globals__)
 def test_shortcircuit_model(self):
  events=[]
  def eq():events.append('eq');return False
  def finite():events.append('finite');return False
  with self.assertRaises(AssertionError):assert False and eq()
  self.assertEqual(events,[])
  with self.assertRaises(AssertionError):
   assert finite()
   assert eq(),'Nonzero initial state outside qualified scope'
  self.assertEqual(events,['finite'])
 def test_phase_prerequisite(self):
  c=types.SimpleNamespace(failed=None);q=Checks(c)
  with self.assertRaises(AssertionError):q.start('replace')
  q.start('shadow')
  with self.assertRaises(AssertionError):q.start('shadow')
 def test_gate_rejects_missing(self):
  with self.assertRaises((KeyError,AssertionError)):eq_gate({},'replace')
 def test_gate_408(self):
  rows=[]
  for site,n in LIMITS.items():
   for i in range(n):rows.append(dict(site=site,index=i+1,mode='replace',input_owner_preserved=True,metadata_preserved=True,same_stream=True,tile=True,native=None,constant=0 if site=='gdn.initial'else 1,physical_elements=16777216 if site=='gdn.initial'else 1))
  r=dict(mode='replace',calls=408,descale384=384,initial24=24,counts=LIMITS,rows=rows,native_predicates=0,tile_predicates=408,shadow_pass=True)
  self.assertTrue(eq_gate(r,'replace'));rows[-1]['tile']=False
  with self.assertRaises(AssertionError):eq_gate(r,'replace')
 def test_ast_and_import_scope(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
  self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda',sys.modules)
if __name__=='__main__':unittest.main()
