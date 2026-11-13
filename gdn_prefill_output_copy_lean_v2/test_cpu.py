import unittest,types,sys
from unittest.mock import patch
from .adapter import Adapter
from gdn_prefill_output_copy_v3.test_cpu import Tensor
class Copy:
 def __init__(self):self.pending=[];self.failed=False;self.calls=0;self.recorded=False
 def __call__(self,d,s):self.calls+=1;self.recorded=True;return d
 def release_after_sync(self):self.pending.clear()
class Tests(unittest.TestCase):
 def tensors(self):
  d,s=Tensor(100),Tensor(20000000);s.device=d.device
  return d,s,types.SimpleNamespace(Tensor=Tensor,bfloat16='bf16',cuda=types.SimpleNamespace(current_stream=lambda *_:types.SimpleNamespace(cuda_stream=1)))
 def test_async_census_release(self):
  a=Adapter();d,s,t=self.tensors();a.layers=set(range(24));a.seen=set(a.layers)
  with patch('gdn_prefill_output_copy_lean_v2.adapter.Copy',Copy):
   for i in range(24):
    a.context=dict(layer=i,destination=d,count=0);self.assertIs(a.observe(t,'aten.copy_.default',(d,s),{}),d)
  a.context=None;self.assertFalse(a.finish()['per_call_sync']);self.assertEqual(len(a.pending),24)
  def bad():raise RuntimeError('sync failure')
  with self.assertRaises(RuntimeError):a.synchronize_and_release(bad)
  self.assertEqual(len(a.pending),24);self.assertTrue(all(not c.pending for c in a.pending))
  receipt=a.synchronize_and_release(lambda:None);self.assertTrue(receipt['copy_owners_released']);self.assertEqual(a.pending,[])
 def test_unqualified_descriptor_latches(self):
  a=Adapter();d,s,t=self.tensors();s.offset=1;s.size+=2;a.context=dict(layer=0,destination=d,count=0)
  with self.assertRaises(ValueError):a.observe(t,'aten.copy_.default',(d,s),{})
  self.assertTrue(a.failed);self.assertEqual(a.pending,[])
 def test_installed_chain_and_close(self):
  a=Adapter();d,s,t=self.tensors();events=[]
  class Mode:
   def __enter__(self):return self
   def __exit__(self,*args):pass
  mods={}
  for name in('torch','torch.utils','torch.utils._python_dispatch','vllm','vllm.forward_context','gdn_post_conv_runtime_capture_v1.observer'):mods[name]=types.ModuleType(name)
  mods['torch.utils._python_dispatch'].TorchDispatchMode=Mode
  mods['vllm.forward_context'].get_forward_context=lambda:types.SimpleNamespace(attn_metadata={str(i):None for i in range(24)})
  mods['gdn_post_conv_runtime_capture_v1.observer'].phase_meta=lambda _:dict(phase='prefill',speculative=False,num_actual_tokens=1696)
  def core(mixed_qkv,b,aa,core_attn_out):
   events.append('existing-v5-chain');a.context['count']+=1;return 'unchanged'
  owners=[types.SimpleNamespace(prefix=str(i),chunk_gated_delta_rule=object(),_forward_core=core)for i in range(24)]
  model=types.SimpleNamespace(named_modules=lambda:((str(i)+'.linear_attn',m)for i,m in enumerate(owners)))
  with patch.dict(sys.modules,mods):
   a.install(model)
   for m in owners:self.assertEqual(m._forward_core(None,None,None,d),'unchanged')
   a.close()
  self.assertEqual(events,['existing-v5-chain']*24);self.assertTrue(all(m._forward_core is core for m in owners))
if __name__=='__main__':unittest.main()
