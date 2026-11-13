import unittest,types,tempfile,ast
from pathlib import Path
from unittest.mock import patch
from lean26_prefill_composition_v1.prefill import check_owned_bridge,PrefillInputs
from lean26_prefill_composition_v1.adapter import SPECS,ORDER
from lean_public_composition_v2.attention_runner import owned_copy
from lean_public_composition_v1.adapter import Controller
R=Path(__file__).parent
class Tests(unittest.TestCase):
 def test_counts_order(self):
  self.assertEqual(len(SPECS),26);self.assertEqual(ORDER[0],'prepare_prefill_inputs');self.assertEqual(sum(n for k,c,n in SPECS),5889)
 def test_real_owned_copy_snapshots_binding(self):
  specdir=R.parent/'runner_padding_runtime_integration_v1';originaltext=(specdir/'prepare_inputs.original.py.txt').read_text();ns={'prepare_prefill_inputs':object()}
  exec('def source(self):pass',ns);original=ns['source'];expected=ns['prepare_prefill_inputs']
  with patch('inspect.getsource',return_value=originaltext):copied=owned_copy(original,'runner_padding_runtime_integration_v1','prepare_inputs',{'_construct':lambda:None,'_fill_padding':lambda:None})
  self.assertIs(copied.__globals__['prepare_prefill_inputs'],expected)
  ns['prepare_prefill_inputs']=object();self.assertIs(copied.__globals__['prepare_prefill_inputs'],expected)
  def wrapper_factory(copied):
   def new(*args):return copied(*args)
   def guarded(*args):return new(*args)
   guarded.__wrapped__=new;return guarded
  part=types.SimpleNamespace(bindings=[(object(),'prepare_inputs',None,None,None,wrapper_factory(copied))])
  self.assertTrue(check_owned_bridge(part,expected))
  with self.assertRaises(AssertionError):check_owned_bridge(part,ns['prepare_prefill_inputs'])
 def test_reverse_restore(self):
  c=Controller(None,specs=());m=types.SimpleNamespace(helper=lambda:1);old=m.helper;p=PrefillInputs(c,'prepare_prefill_inputs',1);p.bind(m,'helper',lambda:2);p.close();self.assertIs(m.helper,old)
 def test_latch_prevents_entry(self):
  c=Controller(None,specs=());c.active=True;p=PrefillInputs(c,'prepare_prefill_inputs',1);c.latch('other',ValueError('first'))
  with self.assertRaises(AssertionError):p.invoke()
 def test_prerequisites(self):
  from .gates import qualification
  self.assertTrue(qualification(R)['actual26_pending'])
 def test_ast(self):
  for root in [R,R.parent/'lean26_prefill_composition_v1']:
   for p in root.glob('*.py'):ast.parse(p.read_text())

class InvokeTests(unittest.TestCase):
 def test_candidate_once_original_identity_and_lifetime(self):
  class T:
   def data_ptr(self):return id(self)
   def untyped_storage(self):return self
   def storage_offset(self):return 0
  c=Controller(None,specs=());p=PrefillInputs(c,'prepare_prefill_inputs',1);c.parts=[p];c.start();args=tuple(T()for _ in range(7));seen=[]
  with patch('lean26_prefill_composition_v1.prefill.prepare_prefill_inputs',lambda *a:seen.append(a)):
   self.assertIsNone(p.invoke(*args));self.assertEqual(p.finish()['calls'],1)
   with self.assertRaises(AssertionError):p.invoke(*args)
  self.assertEqual(len(seen),1);self.assertTrue(all(a is b for a,b in zip(seen[0],args)));self.assertEqual(c.owner_refs,list(args))
 def test_candidate_error_latches(self):
  class T:
   def data_ptr(self):return id(self)
   def untyped_storage(self):return self
   def storage_offset(self):return 0
  c=Controller(None,specs=());p=PrefillInputs(c,'prepare_prefill_inputs',1);c.parts=[p];c.start();args=tuple(T()for _ in range(7))
  with patch('lean26_prefill_composition_v1.prefill.prepare_prefill_inputs',side_effect=ValueError('candidate')) as fn:
   with self.assertRaises(ValueError):p.invoke(*args)
   with self.assertRaises(AssertionError):p.invoke(*args)
   self.assertEqual(fn.call_count,1);self.assertIsNotNone(c.failed)
