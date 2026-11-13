import sys,unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from runtime_metadata_five_lean_v1.controller import Controller,LIMITS
class Tensor:
 def __init__(self,p=100,n=32,dtype='torch.int32'):self.p=p;self.shape=(n,);self.dtype=dtype;self.device='cuda:0';self.requires_grad=False
 def stride(self):return (1,)
 def storage_offset(self):return 0
 def data_ptr(self):return self.p
 def untyped_storage(self):return self
 def nbytes(self):return self.shape[0]*4
 def is_neg(self):return False
 def is_conj(self):return False
class Tests(unittest.TestCase):
 def make(self):return Controller(lambda d:SimpleNamespace(cuda_stream=7))
 def test_counts(self):
  q=self.make();q.start();a=Tensor();counter=[1000]
  def alloc(n=32,dtype='torch.int32'):counter[0]+=128;return Tensor(counter[0],n,dtype)
  for label,n in LIMITS.items():
   for _ in range(n):q.invoke(label,()if label=='full'else(a,),lambda:alloc(2048 if label=='full'else 32,'torch.bool'if label=='gt'else'torch.int32'),'cuda:0','torch.bool'if label=='gt'else'torch.int32',(2048 if label=='full'else 32,))
  result=q.finish();self.assertEqual(result['calls'],5);self.assertFalse(result['raw_rechecked']);self.assertEqual(result['native_oracle_calls'],0)
 def test_alias(self):
  q=self.make();q.start();a=Tensor()
  with self.assertRaises(ValueError):q.invoke('subq',(a,),lambda:a,'cuda:0','torch.int32',(32,))
  self.assertTrue(q.failed)
 def test_mutation(self):
  q=self.make();q.start();a=Tensor()
  def bad():a.p=42;return Tensor(300)
  with self.assertRaises(ValueError):q.invoke('subq',(a,),bad,'cuda:0','torch.int32',(32,))
 def test_cap_before_body(self):
  q=self.make();q.start();q.counts['subq']=1;called=[]
  with self.assertRaises(RuntimeError):q.invoke('subq',(),lambda:called.append(1),'cuda:0','torch.int32',(32,))
  self.assertEqual(called,[])
 def test_stream(self):
  q=self.make();q.start();calls=[0]
  def stream(d):calls[0]+=1;return SimpleNamespace(cuda_stream=calls[0])
  q.current_stream=stream
  with self.assertRaises(ValueError):q.invoke('subq',(),lambda:Tensor(),'cuda:0','torch.int32',(32,))
 def test_finish_latch(self):
  q=self.make();q.start()
  with self.assertRaises(RuntimeError):q.finish()
  called=[]
  with self.assertRaises(RuntimeError):q.guard(lambda:called.append(1))()
  self.assertFalse(called)
 def test_close(self):
  q=self.make();calls=[];q.binding=SimpleNamespace(close=lambda:calls.append(1));q.close();self.assertEqual(calls,[1])
  with self.assertRaises(RuntimeError):q.start()
if __name__=='__main__':unittest.main()
