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

 def test_failure_latch_blocks_every_entry(self):
  for boundary in ('contract','native','candidate'):
   with tempfile.TemporaryDirectory() as td:
    a=adapter.Adapter(Path(td)/'out');events=[]
    @adapter.guarded
    def dispatch(self):
     events.append(boundary);raise ValueError(boundary)
    with self.assertRaises(ValueError):dispatch(a)
    first=a.first_failure.copy()
    for operation in (lambda:dispatch(a),lambda:a.start('shadow'),lambda:a.finish(),lambda:a.apply('x',{},lambda:events.append('native'),{})):
     with self.assertRaises(AssertionError):operation()
    self.assertEqual(events,[boundary]);self.assertEqual(a.first_failure,first)
    self.assertTrue((a.out/'first_mismatch/FIRST-FAILURE.json').is_file())
 def test_snapshot_contract(self):
  s=(Path(__file__).parent/'adapter.py').read_text()
  for key in ('input_','native_','candidate_','320*1024*1024','self.guard()','@guarded'):
   self.assertIn(key,s)

 def test_real_wrapper_failure_latches_before_queued_entry(self):
  for failure in ('predispatch','native','candidate'):
   calls.clear();named={};md={};events=[]
   for i in range(24):
    prefix=f'model.layers.{i}.linear_attn';m=NS(_forward_method=forward_native,gdn_prefill_backend='triton')
    named[prefix]=NS(prefix=prefix);named[prefix+'.chunk_gated_delta_rule']=m
    md[prefix]=NS(num_prefills=32,num_decodes=0,num_actual_tokens=1696,spec_sequence_masks=None)
   with tempfile.TemporaryDirectory() as td,patch.dict(sys.modules,{'vllm.forward_context':NS(get_forward_context=lambda:NS(attn_metadata=md))}):
    a=adapter.Adapter(Path(td)/'out');a.install(NS(named_modules=lambda:named.items()))
    def bad_apply(name,v,native,phase):
     events.append('apply')
     if failure=='candidate':native();events.append('candidate')
     raise RuntimeError(failure)
    a.apply=bad_apply
    if failure=='predispatch':md['model.layers.0.linear_attn'].num_actual_tokens=1
    wrapper=named['model.layers.0.linear_attn.chunk_gated_delta_rule']._forward_method
    with self.assertRaises((RuntimeError,AssertionError)):wrapper()
    before=(list(events),list(calls));first=a.first_failure.copy()
    for _ in range(2):
     with self.assertRaises(AssertionError):wrapper()
    self.assertEqual(before,(events,calls));self.assertEqual(first,a.first_failure);a.close()
 def test_snapshot_saved_bounded_inputs_outputs(self):
  import torch
  with tempfile.TemporaryDirectory() as td:
   a=adapter.Adapter(Path(td)/'out');x=torch.tensor([1.0,-0.0]);a.snapshot('input_q',x);x[0]=2
   a.snapshot('native_o',torch.tensor([3.]));a.snapshot('candidate_o',torch.tensor([4.]))
   a.latch('test',ValueError('first'))
   saved=torch.load(a.out/'first_mismatch/input_q.pt',weights_only=True)
   self.assertEqual(saved[0].item(),1);self.assertEqual(len(a.first_failure['artifacts']),3)

 def test_v3_only_solver_binding_changed(self):
  old=(Path(__file__).parent.parent/'gdn_prefill_core_runtime_integration_v2/adapter.py').read_text()
  new=(Path(__file__).parent/'adapter.py').read_text()
  self.assertEqual(old.replace('_tileir_gdn_prefill_core_integration_v2','_tileir_gdn_prefill_core_integration_v3').replace('from tileir_gdn_solve_tril_v5 import solve_tril','from tileir_gdn_solve_tril_v6 import solve_tril'),new)
