import ast,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import adapter as a
class M:
 attn_output_gate=True
 def __init__(self):self.order=[]
 def forward(self,*args):raise AssertionError('Original forward must not run duplicate gate')
 def qkv_proj(self,x):self.order.append('qkv');return ('qkv',None)
 def _project_qkv_gate(self,q,p):self.order.append('project');return ('q','k','v','gate')
 def attn(self,q,k,v):self.order.append('attention');return 'attention_result'
 def o_proj(self,x):self.order.append(('out',x));return ('result',None)
class Tests(unittest.TestCase):
 def test_forward_order_restore(self):
  mods=[(f'model.layers.{i}.self_attn',M()) for i in range(3,32,4)]
  for _,m in mods:
   m.attn=types.SimpleNamespace(forward=lambda:None,__call__=m.attn)
   # Python special method lookup requires callable class rather than instance.
   class Call:
    def __init__(self,fn):self.fn=fn
    def forward(self):pass
    def __call__(self,*args):return self.fn(*args)
   m.attn=Call(m.attn.__call__);m.o_proj=Call(m.o_proj)
  obs=a.Adapter.__new__(a.Adapter);obs.saved=[];obs.installed=[];obs.apply_gate=lambda name,x,g:'tile_output'
  model=types.SimpleNamespace(named_modules=lambda:mods)
  with patch.object(a,'identity',return_value={'qualname':'Qwen3NextAttention.forward','source_sha256':a.SOURCE_SHA}):obs.install(model)
  self.assertEqual(mods[0][1].forward('positions','hidden'),'result')
  self.assertEqual(mods[0][1].order,['qkv','project','attention',('out','tile_output')])
  obs.close();self.assertTrue(all('forward' not in m.__dict__ for _,m in mods))
 def test_fail_closed(self):
  obs=a.Adapter.__new__(a.Adapter);obs.shadow_pass=False
  with self.assertRaises(AssertionError):obs.start('replace')
 def test_counts(self):
  obs=a.Adapter.__new__(a.Adapter);obs.installed=[{'path':str(i)} for i in range(8)];obs.rows=[dict(path=str(i),shape=[m,4096],passed=True) for i in range(8) for m in [1696]+[32]*15];obs.mode='shadow';obs.native_gate_calls=obs.tileir_gate_calls=128;obs.save=lambda:None;obs.summary=lambda:{'shadow_pass':obs.shadow_pass}
  self.assertTrue(obs.finish()['shadow_pass']);obs.rows.pop()
  with self.assertRaises(AssertionError):obs.finish()
 def test_cpu_ast(self):
  self.assertNotIn('torch',sys.modules);self.assertNotIn('vllm',sys.modules)
  for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
if __name__=='__main__':unittest.main()
