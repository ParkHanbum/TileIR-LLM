import unittest,types,sys,ast,json
from pathlib import Path
from unittest.mock import patch
from .cast import Cast
from .gates import cast_gate
R=Path(__file__).parent
class Tensor:
 def __init__(self,n,dtype='torch.int32',values=None):
  self.shape=(n,);self.dtype=dtype;self.layout='torch.strided';self.device=types.SimpleNamespace(type='cuda');self.requires_grad=False;self.values=list(values or range(n));self.storage=types.SimpleNamespace(nbytes=lambda:n*(4 if dtype=='torch.int32'else 8),data_ptr=lambda:id(self));self.nativecalls=0
 def stride(self):return (1,)
 def storage_offset(self):return 0
 def is_neg(self):return False
 def is_conj(self):return False
 def data_ptr(self):return id(self)
 def untyped_storage(self):return self.storage
 def cpu(self):return list(self.values)
 def view(self,dtype):return self.values
 def long(self):self.nativecalls+=1;out=Tensor(self.shape[0],'torch.int64',self.values);out.device=self.device;return out
class T(unittest.TestCase):
 def part(self):
  p=Cast.__new__(Cast);p.output=R;p.active=False;p.failed=None;p.mode=None;p.rows=[];p.shadow_pass=False;p.owner=None;p.wrapped=None;return p
 def test_actual_cast_body_shadow16_then_replace16(self):
  p=self.part();inputs=[];native_outputs=[];calls=[]
  class Snapshot:
   def set_(self,storage,*a):self.s=storage;return self
   def cpu(self):return ('bytes',self.s.data_ptr(),self.s.nbytes())
  torch=types.SimpleNamespace(uint8='u8',int64='torch.int64',empty=lambda *a,**k:Snapshot(),equal=lambda a,b:a==b)
  def candidate(x):calls.append(1);out=Tensor(x.shape[0],'torch.int64',x.values);out.device=x.device;return out
  fake={'torch':torch,'tileir_vllm_embedding_int64_v1.api':types.SimpleNamespace(embedding_ids_long=candidate)}
  with patch.dict(sys.modules,fake):
   for mode in ('shadow','replace'):
    p.start(mode)
    for i in range(16):
     x=Tensor(1696 if i==0 else 32);inputs.append(x);out=p.cast(x)
     self.assertEqual(out.values,x.values);self.assertEqual(x.nativecalls,1 if mode=='shadow'else 0)
    cast_gate(p.finish(),mode)
  self.assertEqual(len(calls),32)
 def test_owned_source_current_delegate_and_restore(self):
  ns={};exec((R/'forward.original.py.txt').read_text(),ns)
  V=type('VocabParallelEmbedding',(),{'forward':ns['forward']});owner=V();owner.tp_size=1;events=[];owner.quant_method=types.SimpleNamespace(embedding=lambda own,x:events.append(x)or x)
  p=self.part();p.model=types.SimpleNamespace(named_modules=lambda:[('model.embed_tokens',owner)])
  fake={'vllm.model_executor.layers.vocab_parallel_embedding':types.SimpleNamespace(VocabParallelEmbedding=V)}
  p.cast=lambda x:events.append('cast')or 'new_ids'
  with patch.dict(sys.modules,fake),patch('inspect.getsource',return_value=(R/'forward.original.py.txt').read_text()):p.install()
  p.start('shadow')
  self.assertEqual(owner.forward('input'),'new_ids');self.assertEqual(events,['cast','new_ids'])
  owner.quant_method.embedding=lambda own,x:('new_current_delegate',x)
  self.assertEqual(owner.forward('input'),('new_current_delegate','new_ids'))
  p.rows=[{}]*16;events.clear()
  with self.assertRaises(AssertionError):owner.forward('input')
  self.assertEqual(events,[]);self.assertIsNotNone(p.failed);self.assertTrue(p.close()['restored'])
 def test_conditional_and_source_only(self):
  p=self.part()
  with self.assertRaises(AssertionError):p.start('replace')
  self.assertEqual((R/'forward.candidate.py.txt').read_text(),(R/'forward.original.py.txt').read_text().replace('input_.long()','_cast_ids(input_)'))
  for f in R.glob('*.py'):ast.parse(f.read_text())
  self.assertFalse((R/'cache_v1').exists());self.assertFalse((R/'CACHE-PRESEED-01.json').exists())
  for n in ('torch','vllm','cuda.tile'):self.assertNotIn(n,sys.modules)
if __name__=='__main__':unittest.main()
