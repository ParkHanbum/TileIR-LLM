import unittest,types,tempfile
from pathlib import Path
from unittest.mock import patch
from public_vllm_composed_integration_v2 import adapter
from public_vllm_composed_integration_v2.gates import qualification
class Tests(unittest.TestCase):
 def parts(self,fail=None):
  events=[];parts=[]
  for key,n in [('norm',1040),('attention_gate',128),('block_gather',16),('slot_mapping',16),('pos_seq',16)]:
   def install(m,p,key=key):
    events.append(('install',key))
    if key==fail:raise RuntimeError('injected')
    return {}
   def close(m,key=key):events.append(('close',key));return {'restored':True}
   module=types.SimpleNamespace(install=install,close=close,start=lambda m,mode:{},finish=lambda m,n=n:{'shadow_pass':True,'supported_calls':n})
   parts.append((key,module,n))
  return parts,events
 def test_compose(self):
  parts,e=self.parts();m=types.SimpleNamespace()
  with patch.object(adapter,'PARTS',parts),tempfile.TemporaryDirectory() as d:
   adapter.install(m,str(Path(d)/'new'));adapter.start(m,'shadow');r=adapter.finish(m);self.assertEqual(r['supported_calls'],1216);self.assertTrue(r['shadow_pass']);self.assertTrue(adapter.close(m)['restored']);self.assertEqual(e[-5:],[('close','pos_seq'),('close','slot_mapping'),('close','block_gather'),('close','attention_gate'),('close','norm')])
 def test_partial_install_cleanup(self):
  parts,e=self.parts('attention_gate');m=types.SimpleNamespace()
  with patch.object(adapter,'PARTS',parts),tempfile.TemporaryDirectory() as d:
   with self.assertRaises(RuntimeError):adapter.install(m,str(Path(d)/'new'))
  self.assertFalse(hasattr(m,adapter.ATTR));self.assertEqual(e[-1],('close','norm'))
 def test_partial_install_every_boundary(self):
  keys=['norm','attention_gate','block_gather','slot_mapping','pos_seq']
  for i,key in enumerate(keys):
   parts,e=self.parts(key);m=types.SimpleNamespace()
   with patch.object(adapter,'PARTS',parts),tempfile.TemporaryDirectory() as d:
    with self.assertRaises(RuntimeError):adapter.install(m,str(Path(d)/'new'))
   self.assertFalse(hasattr(m,adapter.ATTR))
   self.assertEqual([k for op,k in e if op=='close'],list(reversed(keys[:i])))
 def test_nested_prepare_attn_restoration(self):
  from block_gather_runtime_integration_v1 import adapter as bg
  from slot_mapping_runtime_integration_v1 import adapter as sm
  class Blocks:
   def gather_block_tables(self,*a,**k):return 'gather'
   def compute_slot_mappings(self,*a,**k):return 'slots'
  b=types.SimpleNamespace(idx_mapping=object(),num_reqs_after_padding=32,query_start_loc=object(),positions=object(),num_tokens_after_padding=1696)
  class Runner:
   pcp_manager=None
   def __init__(self):self.block_tables=Blocks()
   def prepare_attn(self,b):
    return (self.block_tables.gather_block_tables(b.idx_mapping,b.num_reqs_after_padding),self.block_tables.compute_slot_mappings(b.idx_mapping,b.query_start_loc,b.positions,b.num_tokens_after_padding))
  orig=Runner.prepare_attn
  with tempfile.TemporaryDirectory() as d:
   a=bg.Adapter(str(Path(d)/'bg'));c=sm.Adapter(str(Path(d)/'sm'))
   a.apply=lambda owner,idx,padded,native,batch:native();c.apply=lambda owner,idx,starts,pos,padded,native,batch:native()
   with patch.object(bg,'phase_metadata',return_value={'num_reqs_after_padding':32}),patch.object(sm,'phase_metadata',return_value={}):
    a.install_classes(Runner,Blocks);inner=Runner.prepare_attn;c.install_classes(Runner,Blocks)
    self.assertEqual(Runner().prepare_attn(b),('gather','slots'));c.close();self.assertIs(Runner.prepare_attn,inner);a.close();self.assertIs(Runner.prepare_attn,orig)
 def test_qualification(self):self.assertEqual(len(qualification(Path(__file__).parent)['components']),5)
if __name__=='__main__':unittest.main()
