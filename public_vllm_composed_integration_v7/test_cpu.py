import unittest,types,tempfile
from pathlib import Path
from unittest.mock import patch
from public_vllm_composed_integration_v7 import adapter
from public_vllm_composed_integration_v7.gates import qualification
class Tests(unittest.TestCase):
 def parts(self,fail=None):
  events=[];parts=[]
  for key,n in [('norm',1040),('attention_gate',128),('block_gather',16),('slot_mapping',16),('pos_seq',16),('gated_norm',384),('sample_counts',16),('qk_rope',128),('silu_mul',512),('indexed_decode_conv',360)]:
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
   adapter.install(m,str(Path(d)/'new'));adapter.start(m,'shadow');r=adapter.finish(m);self.assertEqual(r['supported_calls'],2616);self.assertTrue(r['shadow_pass']);self.assertTrue(adapter.close(m)['restored']);self.assertEqual(e[-10:],[('close','indexed_decode_conv'),('close','silu_mul'),('close','qk_rope'),('close','sample_counts'),('close','gated_norm'),('close','pos_seq'),('close','slot_mapping'),('close','block_gather'),('close','attention_gate'),('close','norm')])
 def test_partial_install_cleanup(self):
  parts,e=self.parts('attention_gate');m=types.SimpleNamespace()
  with patch.object(adapter,'PARTS',parts),tempfile.TemporaryDirectory() as d:
   with self.assertRaises(RuntimeError):adapter.install(m,str(Path(d)/'new'))
  self.assertFalse(hasattr(m,adapter.ATTR));self.assertEqual(e[-1],('close','norm'))
 def test_partial_install_every_boundary(self):
  keys=['norm','attention_gate','block_gather','slot_mapping','pos_seq','gated_norm','sample_counts','qk_rope','silu_mul','indexed_decode_conv']
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
 def test_release_blocked_without_sampler_proof(self):
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(AssertionError):qualification(Path(d))


class GateTests(unittest.TestCase):
 def fixture(self):
  import json
  p=Path(__file__).resolve().parent.parent
  r=json.loads((p/'public_vllm_composed_integration_v6/composedNine01/RESULT.json').read_text())
  conv=json.loads((p/'indexed_decode_conv_runtime_v1/indexedDecodeIntegration01/RESULT.json').read_text())
  for arm in ('shadow_request','replacement_request'):
   d=r[arm]['selected_dispatch'];d['components']['indexed_decode_conv']=conv[arm]['selected_dispatch'];d['supported_calls']=2616
  return r
 def test_combined_fixture(self):
  from public_vllm_composed_integration_v7.result_gate import audit
  self.assertTrue(audit(self.fixture())['passed'])
 def test_numeric_nonexact_reject(self):
  from public_vllm_composed_integration_v7.result_gate import audit
  r=self.fixture();r['public_logprob_gate']['exact_numeric']=False;self.assertFalse(audit(r)['passed'])
 def test_prefill_count_reject(self):
  from public_vllm_composed_integration_v7.result_gate import audit
  r=self.fixture();d=r['replacement_request']['selected_dispatch']['components']['indexed_decode_conv'];k=next(k for k in d['counts'] if k.endswith(':prefill'));d['counts'][k]+=1;self.assertFalse(audit(r)['passed'])
if __name__=='__main__':unittest.main()
