import unittest,tempfile,sys,ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import observer as o
class Tensor:pass
class Tests(unittest.TestCase):
 def test_selected_dispatch_once_identity_restore(self):
  with tempfile.TemporaryDirectory() as t:
   calls=[];ret=(Tensor(),Tensor())
   def selected(q,initial_state,output_final_state=True):calls.append((q,initial_state));return ret
   mods={};owners={}
   for i in range(24):
    name=f'model.layers.{i}.linear_attn';owner=NS(prefix=name,kv_cache=(Tensor(),Tensor()));child=NS(_forward_method=selected,gdn_prefill_backend='triton');mods[name]=owner;mods[name+'.chunk_gated_delta_rule']=child;owners[name]=NS(num_prefills=32,num_decodes=0,num_actual_tokens=1696,spec_sequence_masks=None)
   model=NS(named_modules=lambda:mods.items());obs=o.Observer(Path(t)/'bank')
   fake=NS(get_forward_context=lambda:NS(attn_metadata=owners))
   with patch.dict(sys.modules,{'vllm.forward_context':fake,'torch':NS(Tensor=Tensor)}),patch.object(o,'metadata',lambda x:{}),patch.object(o,'aliases',lambda x:{}):
    obs.tensor=lambda x,label:dict(logical_sha256='same');obs.install(model)
    for i in range(24):self.assertIs(mods[f'model.layers.{i}.linear_attn.chunk_gated_delta_rule']._forward_method(Tensor(),Tensor()),ret)
    self.assertEqual(len(calls),24);self.assertTrue(obs.finish()['complete'])
    self.assertIs(mods['model.layers.0.linear_attn.chunk_gated_delta_rule']._forward_method,selected)
 def test_memory_guard_precedes_copy(self):
  s=Path(__file__).with_name('observer.py').read_text();self.assertLess(s.index('assert t.untyped_storage().nbytes()'),s.index('cpu=t.detach().cpu()'));self.assertEqual(o.CAP,512*1024*1024)
 def test_ast_no_backend_import(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
  self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main()
