import unittest,types,tempfile
from pathlib import Path
from unittest.mock import patch
from public_vllm_composed_integration_v1 import adapter
from public_vllm_composed_integration_v1.gates import qualification
class Tests(unittest.TestCase):
 def parts(self,fail=None):
  events=[];parts=[]
  for key,n in [('norm',1040),('attention_gate',128),('block_gather',16)]:
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
   adapter.install(m,str(Path(d)/'new'));adapter.start(m,'shadow');r=adapter.finish(m);self.assertEqual(r['supported_calls'],1184);self.assertTrue(r['shadow_pass']);self.assertTrue(adapter.close(m)['restored']);self.assertEqual(e[-3:],[('close','block_gather'),('close','attention_gate'),('close','norm')])
 def test_partial_install_cleanup(self):
  parts,e=self.parts('attention_gate');m=types.SimpleNamespace()
  with patch.object(adapter,'PARTS',parts),tempfile.TemporaryDirectory() as d:
   with self.assertRaises(RuntimeError):adapter.install(m,str(Path(d)/'new'))
  self.assertFalse(hasattr(m,adapter.ATTR));self.assertEqual(e[-1],('close','norm'))
 def test_qualification(self):self.assertEqual(len(qualification(Path(__file__).parent)['components']),3)
if __name__=='__main__':unittest.main()
