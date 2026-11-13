import unittest,types,sys
from unittest.mock import patch
from .bindings import HiddenGather,HiddenOriginal
class Tensor:
 def __init__(self,shape,dtype,nbytes):self.shape=shape;self.dtype=dtype;self.device='cuda';self.nbytes=nbytes
 def stride(self):return (self.shape[-1],1)if len(self.shape)==2 else(1,)
 def storage_offset(self):return 0
 def untyped_storage(self):return types.SimpleNamespace(nbytes=lambda:self.nbytes)
 def detach(self):return self
 def cpu(self):return self
 def tolist(self):return list(range(32))
class T(unittest.TestCase):
 def part(self):
  c=types.SimpleNamespace(active=True,failed=None)
  def latch(key,e):
   if c.failed is None:c.failed=(key,e)
  c.latch=latch;p=HiddenGather(c,'hidden_gather',16);p.setup_finite();p.finite_impl_six=lambda x:None;return p
 def test_actual_owned_hidden_body16_and_external17_target0(self):
  p=self.part();writes=[];checks=[]
  torch=types.SimpleNamespace(bfloat16='bf16',int64='i64')
  def target(x,idx):writes.append(1);return Tensor((32,2560),'bf16',163840)
  fake={'torch':torch,'tileir_vllm_hidden_gather_v1':types.SimpleNamespace(gather_hidden_rows=target),'tileir_vllm_hidden_gather_v1.api':types.SimpleNamespace(validate_metadata=lambda *a:None)}
  with patch.object(HiddenOriginal,'install',return_value={}):p.install(None)
  # Actual copied gather body; only numerical/storage external functions are CPU stubs.
  body=next(c.cell_contents for c in p.gather.__closure__ if callable(c.cell_contents))
  body.__globals__['disjoint']=lambda *a:None
  p.finite_impl_six=lambda x:checks.append(sum(p.counts.values()))
  with patch.dict(sys.modules,fake):
   for i in range(16):
    p.pending='prefill'if i==0 else'decode';p.current=None;n=1696 if i==0 else 32
    x=Tensor((n,2560),'bf16',n*5120);idx=Tensor((32,),'i64',256)
    self.assertIs(p.gather(x,idx),p.current)
   self.assertEqual(checks,list(range(1,17)));self.assertEqual(len(writes),16)
   p.current=None
   with self.assertRaisesRegex(AssertionError,'census exceeded'):p.gather(x,idx)
  self.assertEqual(len(writes),16);self.assertEqual(p.finite_counts['hidden_post'],[16,16]);self.assertIsNotNone(p.controller.failed)
 def test_duplicate_site_latches_and_blocks_next_entry(self):
  p=self.part();targets=[]
  def body():
   p.finite_all('hidden_post',None);p.finite_all('hidden_post',None);targets.append(1)
  with self.assertRaisesRegex(AssertionError,'duplicate'):p.guarded_body(body,('hidden_post',))
  first=p.controller.failed
  with self.assertRaises(AssertionError):p.guarded_body(lambda:targets.append(1),('hidden_post',))
  self.assertIs(p.controller.failed,first);self.assertEqual(targets,[]);self.assertEqual(p.finite_counts['hidden_post'],[1,1])
 def test_site_bound_without_external_count_exhaustion(self):
  p=self.part();p.finite_counts['hidden_post']=[16,16];calls=[];p.finite_impl_six=lambda x:calls.append(1)
  with self.assertRaisesRegex(AssertionError,'site census'):p.guarded_body(lambda:p.finite_all('hidden_post',None),('hidden_post',))
  self.assertEqual(calls,[])
 def test_outside_scope_and_nested_rejected(self):
  p=self.part()
  with self.assertRaisesRegex(AssertionError,'outside owned'):p.finite_all('hidden_post',None)
  p=self.part()
  with self.assertRaisesRegex(AssertionError,'Nested'):p.guarded_body(lambda:p.guarded_body(lambda:None,('hidden_post',)),('hidden_post',))
if __name__=='__main__':unittest.main()
