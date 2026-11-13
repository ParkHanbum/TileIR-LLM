import unittest,ast,tempfile,sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from . import adapter
from .result_gate import audit
calls=[]
def forward_native(q=None,k=None,v=None,g=None,beta=None,initial_state=None,output_final_state=True,cu_seqlens=None,chunk_indices=None,chunk_offsets=None,use_qk_l2norm_in_kernel=False,core_attn_out=None,aiter_prefill_metadata=None):
 calls.append(q);return (q,initial_state)
class Tests(unittest.TestCase):
 def test_wrappers_shadow_replace_restore(self):
  calls.clear();named={};md={}
  for i in [i for i in range(32) if i%4!=3]:
   prefix=f'model.layers.{i}.linear_attn';owner=NS(prefix=prefix);m=NS(_forward_method=forward_native,gdn_prefill_backend='triton');named[prefix]=owner;named[prefix+'.chunk_gated_delta_rule']=m;md[prefix]=NS(num_prefills=32,num_decodes=0,num_actual_tokens=1696,spec_sequence_masks=None)
  model=NS(named_modules=lambda:named.items())
  with tempfile.TemporaryDirectory() as td,patch.dict(sys.modules,{'vllm.forward_context':NS(get_forward_context=lambda:NS(attn_metadata=md))}):
   a=adapter.Adapter(Path(td)/'out');a.install(model)
   def apply(name,v,native,phase):
    if a.mode=='shadow':ret=native();a.native_gate_calls+=1
    else:ret=(v['q'],v['initial_state'])
    a.tileir_gate_calls+=1;a.rows.append(dict(path=name,passed=True));return ret
   a.apply=apply
   with self.assertRaises(AssertionError):a.start('replace')
   for mode in ('shadow','replace'):
    a.start(mode)
    for n,m in named.items():
     if n.endswith('chunk_gated_delta_rule'):
      q=object();state=object();ret=m._forward_method(q=q,initial_state=state);self.assertIs(ret[0],q);self.assertIs(ret[1],state)
    self.assertTrue(a.finish()['shadow_pass'])
   self.assertEqual(len(calls),24);a.close()
   self.assertTrue(all(m._forward_method is forward_native for n,m in named.items() if n.endswith('chunk_gated_delta_rule')))
 def test_partial_gate(self):self.assertFalse(audit({})['passed'])
 def test_source_scope(self):
  s=(Path(__file__).parent/'adapter.py').read_text();self.assertIn('Nonzero initial state outside qualified',s);self.assertIn("before==self.baselines[name]['inputs']",s);self.assertNotIn('ssm_state[',s)
 def test_ast(self):
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
