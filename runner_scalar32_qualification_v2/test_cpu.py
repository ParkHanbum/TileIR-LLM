import ast,__future__,unittest,struct
from pathlib import Path
from runner_scalar32_qualification_v2.binding import EXPECTED
from runner_scalar32_qualification_v2.qualifier import ScalarQualifier,owner_signature,synthetic
class V:
 shape=(32,);dtype='torch.int32';device='cuda:0';requires_grad=False
 def __init__(self,values=None):self.v=list(values or range(32))
 def stride(self):return(1,)
 def storage_offset(self):return 0
 def element_size(self):return 4
 def untyped_storage(self):return self
 def nbytes(self):return 128
 def data_ptr(self):return id(self)*4
 def clone(self):return V(self.v)
 def __getitem__(self,i):
  parent=self
  class Scalar:
   def fill_(self,v):parent.v[i]=v
  return Scalar()
def raw(v):return struct.pack('<32i',*v.v)
def candidate(v,i):v.v[i]=1
def owner():
 t=ast.parse(Path(EXPECTED['path']).read_text()); c=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name=='MambaHybridModelState');c.body=[n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='add_request']
 class Base:
  def add_request(self,*a):self.base_calls+=1
 ns={'DefaultModelState':Base};exec(compile(ast.Module(body=[c],type_ignores=[]),EXPECTED['path'],'exec',flags=__future__.annotations.compiler_flag),ns)
 x=ns['MambaHybridModelState']();x.max_num_reqs=32;x._align_mode=False;x.recoverssm=None;x.num_accepted_tokens_gpu=V();x.base_calls=0;return x
class Tests(unittest.TestCase):
 def test_full_phases(self):
  x=owner();q=ScalarQualifier(x,candidate,raw);q.binding.install()
  for phase in ['shadow','replace']:
   x.num_accepted_tokens_gpu.v=list(range(32));q.start(phase)
   for i in range(32):x.add_request(i,None)
   self.assertEqual(q.finish()['calls'],32)
  q.close();self.assertEqual(x.base_calls,64)
 def test_synthetic(self):self.assertTrue(synthetic(candidate,raw,V)['passed'])
 def test_align_refused(self):
  x=owner();x._align_mode=True
  with self.assertRaises(ValueError):owner_signature(x)
 def test_owner_swap_before_super(self):
  x=owner();q=ScalarQualifier(x,candidate,raw);q.binding.install();q.start('shadow');x.num_accepted_tokens_gpu=V()
  with self.assertRaises(ValueError):x.add_request(0,None)
  self.assertEqual(x.base_calls,0);q.close()
 def test_latch(self):
  x=owner();q=ScalarQualifier(x,lambda v,i:None,raw);q.binding.install();q.start('shadow')
  with self.assertRaises(ValueError):x.add_request(0,None)
  with self.assertRaises(RuntimeError):x.add_request(1,None)
  self.assertEqual(x.base_calls,1);q.close()
 def test_missing_shadow(self):
  q=ScalarQualifier(owner(),candidate,raw)
  with self.assertRaises(RuntimeError):q.start('replace')
 def test_no_framework(self):
  import sys
  self.assertNotIn('torch',sys.modules);self.assertNotIn('cuda',sys.modules)
if __name__=='__main__':unittest.main()
