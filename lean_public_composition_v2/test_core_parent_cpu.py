"""CPU-only binding/order tests. No tensor kernel execution or GPU query."""
import ast
import inspect
import sys
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
# Import once before patch.dict restores sys.modules; do not unload/reimport Torch.
import torch
P=Path(__file__).resolve().parents[1]
if str(P)not in sys.path:sys.path.insert(0,str(P))
from lean_public_composition_v2.core_parent import PrefillCore,GDNParent,_LeanParentEndpoint
import lean_public_composition_v2.core_parent as cp
import gdn_parent_state_no_oracle_v2.adapter as parent_source

class C:
 def __init__(self):self.active=True;self.failed=None;self.parts=[]
 def latch(self,key,e):
  if self.failed is None:self.failed=(key,repr(e))
class Core:
 gdn_prefill_backend='triton'
 def __init__(self):self._forward_method=self.forward_native;self.native_calls=0
 def forward_native(self,initial_state):self.native_calls+=1;raise AssertionError('Native core forbidden')
class Owner:
 def __init__(self,i):self.prefix=str(i);self.chunk_gated_delta_rule=Core();self._forward_core=self.original
 def original(self,*a,**kw):return None
class Model:
 def __init__(self):self.owners=[Owner(i)for i in range(24)]
 def named_modules(self):
  for i,m in enumerate(self.owners):
   p=f'layers.{i}.linear_attn';yield p,m;yield p+'.chunk_gated_delta_rule',m.chunk_gated_delta_rule
class Tests(unittest.TestCase):
 def setUp(self):
  self.c=C();self.model=Model();ctx=types.SimpleNamespace(attn_metadata={str(i):{}for i in range(24)})
  self.patches=[patch.dict(sys.modules,{'vllm.forward_context':types.SimpleNamespace(get_forward_context=lambda:ctx)}),patch.object(cp,'phase_meta',lambda _:dict(phase='prefill',speculative=False,num_actual_tokens=1696)),patch.object(parent_source,'identity',lambda x:dict(name=x.__name__)),patch.object(parent_source,'metadata',lambda x:{}),patch.object(parent_source,'aliases',lambda x:{})]
  for p in self.patches:p.start()
 def tearDown(self):
  for p in reversed(self.patches):p.stop()
 def setup_parts(self):
  core=PrefillCore(self.c,'gdn_prefill_core',24);self.c.parts.append(core);core.install(self.model)
  parent=GDNParent(self.c,'gdn_parent_state',24);self.c.parts.append(parent);parent.install(self.model)
  return core,parent
 def test_actual_nested_core_binding_and_reverse_restore(self):
  core,parent=self.setup_parts();core.start();parent.start();calls=[]
  core.apply=lambda args:(calls.append(args)or('out','final'))
  for m in self.model.owners:
   initial=object();parent.endpoint.active=dict(initial=initial,owner=object(),row=dict(order=['gather','invert','mask']))
   ret=m.chunk_gated_delta_rule._forward_method(initial)
   self.assertEqual(ret,('out','final'));self.assertEqual(parent.endpoint.active['row']['order'],['gather','invert','mask','core_enter','core_exit'])
   self.assertEqual(m.chunk_gated_delta_rule.native_calls,0)
   parent.endpoint.active=None
  self.assertEqual(len(calls),24);self.assertEqual(core.finish()['calls'],24)
  parent.close();self.assertTrue(all(m._forward_method is fn for m,fn in core.core_bindings.items()))
  core.close();self.assertTrue(all(m.chunk_gated_delta_rule._forward_method==m.chunk_gated_delta_rule.forward_native for m in self.model.owners))
 def test_foreign_core_binding_rejected_before_parent_install(self):
  core=PrefillCore(self.c,'gdn_prefill_core',24);self.c.parts.append(core);core.install(self.model)
  self.model.owners[0].chunk_gated_delta_rule._forward_method=lambda:None
  with self.assertRaises(AssertionError):GDNParent(self.c,'gdn_parent_state',24).install(self.model)
 def test_parent_does_not_create_diagnostic_output_or_spoof_fields(self):
  core,parent=self.setup_parts();e=parent.endpoint
  self.assertFalse(hasattr(e,'out'));self.assertEqual(e.mode,'replace');self.assertNotIn('_composition_entry_key',vars(next(iter(core.core_bindings.values()))))
  self.assertEqual(e.save(),dict(calls=0,target_oracles=0));parent.close();core.close()
 def test_failure_latches_controller(self):
  core,parent=self.setup_parts();parent.endpoint.fail(ValueError('scope'))
  with self.assertRaises(AssertionError):parent.endpoint.guard()
  self.assertEqual(self.c.failed[0],'gdn_parent_state');parent.close();core.close()
 def test_parent_finish_requires_all_four_ops_and_order(self):
  core,parent=self.setup_parts();parent.start();e=parent.endpoint
  e.rows=[dict(layer=n,passed=True,order=['gather','invert','mask','core_enter','core_exit','scatter'],native_target_calls=0,native_invert_calls=0,tile_target_calls=3,tile_invert_calls=1)for n in e.modules]
  self.assertEqual(parent.finish()['logical_suboperation_calls'],96)
  e.rows[0]['native_target_calls']=1
  with self.assertRaises(AssertionError):parent.finish()
  parent.close();core.close()
 def test_source_has_no_external_tensor_diagnostics_or_sync(self):
  source=Path(cp.__file__).read_text();ast.parse(source)
  for x in ('torch.cuda.synchronize','torch.save','hashlib','snapshot(','progress('):self.assertNotIn(x,source)
  self.assertIn("(inputs['initial_state']==0).all()",source)
  self.assertIn('torch.isfinite(t).all()',source)
if __name__=='__main__':unittest.main()
