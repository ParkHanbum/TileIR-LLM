import unittest,tempfile,sys,ast
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import observer as o
calls=[]
def reshape_and_cache_flash(key,value,key_cache,value_cache,slot_mapping,kv_cache_dtype,k_scale,v_scale):
 calls.append((key,value,slot_mapping));return None
class FlashAttentionImpl:
 kv_cache_dtype='auto'
 def do_kv_cache_update(self,layer,key,value,kv_cache,slot_mapping):
  return reshape_and_cache_flash(key,value,kv_cache,kv_cache,slot_mapping,'auto',None,None)
class CPUList:
 def __init__(self,x):self.x=x
 def detach(self):return self
 def cpu(self):return self
 def tolist(self):return self.x
class Tests(unittest.TestCase):
 def test_phase(self):
  self.assertEqual(o.phase_from_lists(list(range(0,1697,53)),[53]*32),'prefill')
  self.assertEqual(o.phase_from_lists(list(range(33)),[54]*32),'decode')
  with self.assertRaises(AssertionError):o.phase_from_lists(list(range(33)),[53]*32)
 def test_slots(self):
  self.assertEqual(o.valid_slots([0,2],3),[0,2])
  for x in ([-1],[3],[1,1]):
   with self.assertRaises(AssertionError):o.valid_slots(x,3)
 def test_real_wrappers_exact_once_and_restore(self):
  calls.clear();mods={};md={}
  for i in range(3,32,4):
   n=f'model.layers.{i}.self_attn.attn';mods[n]=NS(impl=FlashAttentionImpl(),kv_cache=object(),layer_name=n);md[n]=NS(query_start_loc=CPUList(list(range(0,1697,53))),seq_lens=CPUList([53]*32))
  model=NS(named_modules=lambda:mods.items());original=reshape_and_cache_flash
  with tempfile.TemporaryDirectory() as t,patch.dict(sys.modules,{'vllm.forward_context':NS(get_forward_context=lambda:NS(attn_metadata=md))}):
   ob=o.Observer(Path(t)/'bank')
   def capture(ctx,v,native,fn):
    ret=native();self.assertIsNone(ret);ob.samples.append(dict(phase=ctx['phase']));return ret
   ob.capture=capture;ob.install(model)
   for step in range(16):
    for n,m in mods.items():
     md[n].query_start_loc=CPUList(list(range(0,1697,53)) if step==0 else list(range(33)));md[n].seq_lens=CPUList([53+step]*32)
     self.assertIsNone(m.impl.do_kv_cache_update(m,object(),object(),m.kv_cache,object()))
   self.assertEqual(len(calls),128);self.assertTrue(ob.finish()['complete']);self.assertIs(reshape_and_cache_flash,original)
   self.assertTrue(all('do_kv_cache_update' not in m.impl.__dict__ for m in mods.values()))
 def test_ast_import(self):
  for p in Path(__file__).parent.glob('*.py'):ast.parse(p.read_text())
  self.assertNotIn('torch',sys.modules)
if __name__=='__main__':unittest.main()
