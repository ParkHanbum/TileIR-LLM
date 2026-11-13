import ast,copy,json,struct,sys,unittest,__future__
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from batch_ptr_int_cast_v1.api import Transport
from batch_ptr_int_cast_v1.contract import low32
from batch_ptr_int_cast_v1 import binding
class Storage:
 next=4096
 def __init__(self,size):self.size=size;self.ptr=Storage.next;Storage.next+=16384;self.data=bytearray([0xA5]*size)
 def nbytes(self):return self.size
 def data_ptr(self):return self.ptr
class Device:
 def __init__(self,kind):self.type=kind
 def __str__(self):return self.type+(':0'if self.type=='cuda'else'')
 def __eq__(self,o):return type(o)is Device and self.type==o.type
class Tensor:
 def __init__(self,kind='cuda',dtype='torch.int32',size=8192,pinned=True):self.device=Device(kind);self.dtype=dtype;self.shape=(224,);self.requires_grad=False;self.s=Storage(size);self.pinned=pinned
 def stride(self):return (1,)
 def storage_offset(self):return 0
 def data_ptr(self):return self.s.ptr
 def untyped_storage(self):return self.s
 def is_pinned(self):return self.pinned
 def is_neg(self):return False
 def is_conj(self):return False
 def copy_(self,src,non_blocking):
  assert self.dtype==src.dtype=='torch.int64' and non_blocking is True
  self.s.data[:]=src.s.data;return self
class Test(unittest.TestCase):
 def fixture(self):
  src=Tensor('cpu','torch.int64',1792);dst=Tensor();vals=([-2**63,2**63-1,-2**32,2**32,-1,0,1,2**32+17]*28)
  src.s.data[:]=struct.pack('<224q',*vals);stream=SimpleNamespace(cuda_stream=7,synchronize=lambda:None);alloc=[]
  def empty(shape,dtype,device):
   assert shape==(224,) and dtype=='torch.int64' and device==dst.device
   x=Tensor('cuda',dtype,1792);alloc.append(x);return x
  torch=SimpleNamespace(int64='torch.int64',empty=empty,cuda=SimpleNamespace(current_stream=lambda d:stream));calls=[]
  def launch(st,grid,kernel,args):
   calls.append(1);a,b=args;v=struct.unpack('<224q',a.s.data);b.s.data[:896]=struct.pack('<224i',*[low32(x) for x in v])
  ct=SimpleNamespace(launch=launch);transport=Transport(torch,ct,object());return src,dst,transport,stream,alloc,calls
 def test_raw_same_dtype_transport(self):
  src,dst,t,s,a,c=self.fixture();before=bytes(src.s.data);outside=bytes(dst.s.data[896:]);out=t.cast_into(dst,src)
  self.assertIs(out,dst);self.assertEqual(src.s.data,before);self.assertEqual(dst.s.data[896:],outside);self.assertEqual(struct.unpack('<224i',dst.s.data[:896]),tuple(low32(x)for x in struct.unpack('<224q',before)));self.assertEqual(len(t.pending),1);self.assertEqual(len(a),1);self.assertEqual(c,[1]);self.assertEqual(t.ledger[0]['temporary_bytes'],1792)
  t.synchronize_and_release();self.assertFalse(t.pending)
 def test_pending_scope(self):
  src,dst,t,s,a,c=self.fixture();t.cast_into(dst,src)
  with self.assertRaises(RuntimeError):t.cast_into(dst,src)
  self.assertEqual(len(a),1);self.assertEqual(len(c),1);self.assertTrue(t.failed)
 def test_unpinned_preallocation(self):
  src,dst,t,s,a,c=self.fixture();src.pinned=False
  with self.assertRaises(ValueError):t.cast_into(dst,src)
  self.assertFalse(a);self.assertFalse(c);self.assertTrue(t.failed)
 def test_extent_preallocation(self):
  src,dst,t,s,a,c=self.fixture();dst.s.size=8196
  with self.assertRaises(ValueError):t.cast_into(dst,src)
  self.assertFalse(a)
 def test_nonblocking_preallocation(self):
  src,dst,t,s,a,c=self.fixture()
  with self.assertRaises(ValueError):t.cast_into(dst,src,False)
  self.assertFalse(a)
 def test_failed_sync_retains(self):
  src,dst,t,s,a,c=self.fixture();t.cast_into(dst,src)
  def bad():raise RuntimeError('sync')
  s.synchronize=bad
  with self.assertRaises(RuntimeError):t.synchronize_and_release()
  self.assertEqual(len(t.pending),1)
 def test_failed_launch_retains(self):
  src,dst,t,s,a,c=self.fixture()
  def bad(*args):raise RuntimeError('launch')
  t.tile.launch=bad
  with self.assertRaises(RuntimeError):t.cast_into(dst,src)
  self.assertTrue(t.failed);self.assertEqual(len(t.pending),1);self.assertEqual(t.ledger[0]['cast_launch_attempts'],1);self.assertEqual(t.ledger[0]['cast_launch_returns'],0)
 def test_failed_transfer_retains(self):
  src,dst,t,s,a,c=self.fixture();old=t.torch.empty
  def make(*args,**kwargs):
   x=old(*args,**kwargs)
   def bad(*a,**k):raise RuntimeError('copy')
   x.copy_=bad;return x
  t.torch.empty=make
  with self.assertRaises(RuntimeError):t.cast_into(dst,src)
  self.assertEqual(len(t.pending),1);self.assertEqual(t.ledger[0]['transfer_returns'],0);self.assertEqual(t.ledger[0]['cast_launch_attempts'],0)
 def test_actual_source_copy(self):
  e=binding.EXPECTED;tree=ast.parse(Path(e['path']).read_text());node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==e['name']));ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),e['path'],'exec',flags=__future__.annotations.compiler_flag),ns);fn=ns[e['name']];copyfn=binding.build(fn,lambda *a,**k:None);self.assertEqual(copyfn.__defaults__,fn.__defaults__);self.assertEqual(copyfn.__kwdefaults__,fn.__kwdefaults__);self.assertIsNot(copyfn,fn)
if __name__=='__main__':unittest.main()
