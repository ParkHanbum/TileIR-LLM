import ast,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from .binding import Binding
from staged_ownership_runtime_v1.observer import comparator
class T:
 def __init__(self,v,dtype):self.values=v;self.dtype=dtype;self.device='cuda:0'
class Tests(unittest.TestCase):
 def fixture(self):
  native=[];creation=[]
  def tensor(v,**kw):creation.append(kw['dtype']);return T(v,kw['dtype'])
  def equal(a,b):native.append(1);return a.values==b.values
  torch=SimpleNamespace(tensor=tensor,uint64='torch.uint64',int64='torch.int64',equal=equal,cuda=SimpleNamespace(synchronize=lambda:None,current_stream=lambda:7))
  owners=[SimpleNamespace(data_ptr=lambda i=i:100+i,stride=lambda axis:128)for i in range(4)]
  return torch,owners,T([100,101,102,103],torch.uint64),T([128]*4,torch.int64),native,creation
 def test_request_census_native_warmup_only(self):
  torch,owners,p,s,native,creation=self.fixture();phase=['warmup'];obs=SimpleNamespace(compare=comparator(torch));old=obs.compare
  with patch('ownership_table_pair_equal_lean_v1.gates.qualification',return_value=True),patch('ownership_table_pair_equal_lean_v1.callback.equal',side_effect=lambda a,b:a.values==b.values):
   b=Binding(obs,torch,lambda:phase[0]);b.install();self.assertTrue(obs.compare(p,s,owners));self.assertEqual(len(native),2)
   phase[0]='request'
   for _ in range(34):self.assertTrue(obs.compare(p,s,owners))
   r=b.finish();self.assertEqual(r['native_request_equal_calls'],0);self.assertEqual(len(native),2);self.assertEqual(creation,['torch.uint64','torch.int64']*35)
   b.close();self.assertIs(obs.compare,old)
 def test_false_pointer_shortcircuit(self):
  torch,owners,p,s,native,creation=self.fixture();obs=SimpleNamespace(compare=comparator(torch));calls=[]
  with patch('ownership_table_pair_equal_lean_v1.gates.qualification',return_value=True),patch('ownership_table_pair_equal_lean_v1.callback.equal',side_effect=lambda a,b:calls.append(a.dtype)or False):
   b=Binding(obs,torch,lambda:'request');b.install();self.assertFalse(obs.compare(p,s,owners));self.assertEqual(calls,['torch.uint64']);b.close()
 def test_exception_latch(self):
  torch,owners,p,s,_,_=self.fixture();obs=SimpleNamespace(compare=comparator(torch))
  with patch('ownership_table_pair_equal_lean_v1.gates.qualification',return_value=True),patch('ownership_table_pair_equal_lean_v1.callback.equal',side_effect=ValueError('CPU injection'))as eq:
   b=Binding(obs,torch,lambda:'request');b.install()
   with self.assertRaises(ValueError):obs.compare(p,s,owners)
   with self.assertRaises(RuntimeError):obs.compare(p,s,owners)
   self.assertEqual(eq.call_count,1);b.close()
 def test_real_preengine_ast_chain(self):
  from staged_typed_lean_runtime_v1.replacement import OwnershipObserver
  torch,owners,p,s,native,_=self.fixture()
  class Staged:
   def __init__(self):pass
   def apply_write(self):pass
  class Fused:
   def apply(self,*a):pass
  class Tables:
   def init_block_table_layout_tensors(self):pass
  path=Path(__file__).parent.parent/'whole_lean_extensions_runtime_v5/extension.py'
  tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef));node=next(x for x in cls.body if isinstance(x,ast.FunctionDef)and x.name=='pre_engine')
  node.body=[x for x in node.body if not isinstance(x,(ast.Import,ast.ImportFrom))]
  ns=dict(OwnershipObserver=OwnershipObserver,comparator=comparator,verify=lambda *a:{},bm=SimpleNamespace(StagedWriteTensor=Staged,FusedStagedWriter=Fused),bt=SimpleNamespace(BlockTables=Tables),Path=Path,__file__=str(path));exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual pre_engine AST>','exec'),ns)
  ext=SimpleNamespace(phase='setup',bus=SimpleNamespace(failed=None,latch=lambda *a:None));ns['pre_engine'](ext,torch)
  self.assertEqual(ext.phase,'warmup');old=ext.staged.compare
  with patch('ownership_table_pair_equal_lean_v1.gates.qualification',return_value=True),patch('ownership_table_pair_equal_lean_v1.callback.equal',side_effect=lambda a,b:a.values==b.values):
   b=Binding(ext.staged,torch,lambda:ext.phase);b.install();ext.phase='request'
   for _ in range(34):self.assertTrue(ext.staged.compare(p,s,owners))
   self.assertEqual(b.finish()['pair_checks'],68);b.close();self.assertIs(ext.staged.compare,old)
  ext.staged.close();self.assertFalse(ext.staged.patches)
 def test_foreign_close_hold(self):
  torch,*_=self.fixture();obs=SimpleNamespace(compare=comparator(torch))
  with patch('ownership_table_pair_equal_lean_v1.gates.qualification',return_value=True):
   b=Binding(obs,torch,lambda:'request');b.install();foreign=lambda:False;obs.compare=foreign
   with self.assertRaises(RuntimeError):b.close()
   self.assertIs(obs.compare,foreign)
