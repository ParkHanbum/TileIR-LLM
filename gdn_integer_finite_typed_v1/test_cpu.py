import ast,unittest
from pathlib import Path
from types import SimpleNamespace
from .api import all_finite_integer
class Tensor:
 dtype='torch.int32';device=SimpleNamespace(type='cuda');layout='torch.strided';shape=(33,);requires_grad=False
 def __init__(self,n=33):self.n=n;self.shape=(n,)
 def stride(self):return (1,)
 def storage_offset(self):return 0
 def numel(self):return self.n
 def untyped_storage(self):return SimpleNamespace(nbytes=lambda:self.n*(8 if self.dtype=='torch.int64' else 4),data_ptr=lambda:4096)
 def data_ptr(self):return 4096
 def is_neg(self):return False
 def is_conj(self):return False
 def is_contiguous(self):return True
class Tests(unittest.TestCase):
 def test_true_and_empty_sync(self):
  for dtype in ('torch.int32','torch.int64'):
   for n in (0,33):
    t=Tensor(n);t.dtype=dtype;events=[];s=SimpleNamespace(cuda_stream=7,synchronize=lambda:events.append('sync'))
    self.assertIs(all_finite_integer(t,lambda d:s),True);self.assertEqual(events,['sync'])
 def test_wrong_type_before_sync(self):
  t=Tensor();t.dtype='torch.float32';events=[]
  with self.assertRaises(ValueError):all_finite_integer(t,lambda d:events.append(1))
  self.assertFalse(events)
 def test_sync_exception_propagates(self):
  def bad():raise RuntimeError('injected CPU exception')
  with self.assertRaises(RuntimeError):all_finite_integer(Tensor(),lambda d:SimpleNamespace(cuda_stream=7,synchronize=bad))
 def test_stream_change_rejects(self):
  streams=iter([SimpleNamespace(cuda_stream=7,synchronize=lambda:None),SimpleNamespace(cuda_stream=8)])
  with self.assertRaises(ValueError):all_finite_integer(Tensor(),lambda d:next(streams))
 def test_metadata_change_rejects(self):
  t=Tensor()
  def mutate():t.n=34
  with self.assertRaises(ValueError):all_finite_integer(t,lambda d:SimpleNamespace(cuda_stream=7,synchronize=mutate))
 def test_flags_rejected(self):
  t=Tensor();t.requires_grad=True
  with self.assertRaises(ValueError):all_finite_integer(t,lambda d:None)
 def core(self):
  # Compile actual new class without framework-bearing ancestor imports.
  tree=ast.parse(Path(__file__).with_name('core.py').read_text());tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
  class Original:
   def finite(self,t):return 'float-delegate'
  ns=dict(Original=Original,INTEGER=('cu_seqlens','chunk_indices','chunk_offsets'),ORDER=('cu_seqlens','chunk_indices','chunk_offsets','q'),all_finite_integer=all_finite_integer)
  exec(compile(tree,'<CPU typed core>','exec'),ns);obj=object.__new__(ns['PrefillCore']);obj.key='gdn';obj.finite_scope=0;obj.site_counts=dict.fromkeys(ns['INTEGER'],0);obj.descriptors={}
  def latch(key,e):obj.controller.failed=(key,type(e).__name__)
  obj.controller=SimpleNamespace(active=True,failed=None,latch=latch);return obj
 def test_integer_failure_latches(self):
  c=self.core()
  def fail(site,t):raise RuntimeError('injected CPU failure')
  c.integer_check=fail
  with self.assertRaises(RuntimeError):c.finite(Tensor())
  self.assertIsNotNone(c.controller.failed);self.assertEqual(c.site_counts['cu_seqlens'],0)
  with self.assertRaises(AssertionError):c.finite(Tensor())
 def test_integer_cap_and_advance(self):
  c=self.core();c.integer_check=lambda site,t:True;self.assertIs(c.finite(Tensor()),True);self.assertEqual(c.finite_scope,1);self.assertEqual(c.site_counts['cu_seqlens'],1)
  c.finite_scope=0;c.site_counts['cu_seqlens']=24
  with self.assertRaises(AssertionError):c.finite(Tensor())
 def test_float_delegates(self):
  c=self.core();c.finite_scope=3;self.assertEqual(c.finite(Tensor()),'float-delegate')
if __name__=='__main__':unittest.main()
