import unittest,tempfile,types
from pathlib import Path
from unittest.mock import patch
from public_vllm_composed_integration_v9 import adapter
from public_vllm_composed_integration_v8 import adapter as old
from public_vllm_composed_integration_v9.gates import qualification
class Tests(unittest.TestCase):
 def test_existing_order_by_reference(self):
  self.assertEqual(adapter.PARTS[:12],old.PARTS);self.assertEqual(sum(n for k,m,n in adapter.PARTS),2808)
  self.assertEqual([k for k,m,n in adapter.PARTS[-2:]],['embedding','attention_kv'])
 def test_all_partial_install_cleanup(self):
  real=adapter.PARTS
  for fail in range(15):
   events=[];parts=[]
   for i,(key,_,count) in enumerate(real):
    def install(model,out,i=i,key=key):
     events.append(('install',key))
     if i==fail:raise RuntimeError('injected')
     return {}
    def close(model,key=key):events.append(('close',key));return {'restored':True}
    parts.append((key,types.SimpleNamespace(install=install,close=close,start=lambda m,mode:{},finish=lambda m,n=count:dict(supported_calls=n,shadow_pass=True)),count))
   with tempfile.TemporaryDirectory() as d,patch.object(adapter,'PARTS',parts):
    m=types.SimpleNamespace()
    if fail<14:
     with self.assertRaises(RuntimeError):adapter.install(m,str(Path(d)/'out'))
    else:adapter.install(m,str(Path(d)/'out'));adapter.start(m,'shadow');self.assertEqual(adapter.finish(m)['supported_calls'],2808);adapter.close(m)
    self.assertEqual([k for op,k in events if op=='close'],list(reversed([k for k,_,_ in real[:min(fail,14)]])))
 def test_proof_release(self):
  self.assertEqual(len(qualification(Path(__file__).parent)['proofs']),3)
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(AssertionError):qualification(Path(d))
 def test_binding_domains(self):
  p=Path(__file__).parent.parent
  embed=(p/'tp1_embedding_runtime_integration_v1/adapter.py').read_text();kv=(p/'attention_kv_runtime_integration_v2/adapter.py').read_text();qk=(p/'qk_rope_runtime_integration_v1/adapter.py').read_text()
  self.assertIn('q.embedding=wrapper',embed);self.assertNotIn('_project_qkv_gate',embed)
  self.assertIn('impl.do_kv_cache_update=update',kv);self.assertIn('mod.reshape_and_cache_flash=helper',kv);self.assertNotIn('_project_qkv_gate',kv)
  self.assertNotIn('do_kv_cache_update',qk);self.assertNotIn('embedding',qk)
if __name__=='__main__':unittest.main()
