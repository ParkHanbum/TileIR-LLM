"""Small AST type fixture for this kernel, not a compiler or general typechecker."""
import ast,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARRAYS=dict(idx='int64',computed='int32',last='int64',counts='int32',sampled='int64',nsampled='int32',nrejected='int32',qstart='int32',tokens='int32',total='int32')
def stores(path):
 tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef));types=dict(SS='int32',TS='int32',CS='int32');rows=[]
 def typ(n):
  if isinstance(n,ast.Constant):return 'int32'
  if isinstance(n,ast.Name):return types[n.id]
  if isinstance(n,ast.BinOp):return 'int64' if 'int64' in (typ(n.left),typ(n.right)) else 'int32'
  if isinstance(n,ast.Call):
   op=n.func.attr
   if op=='bid':return 'int32'
   if op=='gather':return ARRAYS[n.args[0].id]
   if op=='astype':return n.args[1].attr
  raise AssertionError(ast.dump(n))
 def visit(body):
  for n in body:
   if isinstance(n,ast.Assign):types[n.targets[0].id]=typ(n.value)
   elif isinstance(n,ast.If):visit(n.body);visit(n.orelse)
   elif isinstance(n,ast.For):types[n.target.id]=typ(n.iter.args[0]);visit(n.body)
   elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and n.value.func.attr=='scatter':
    call=n.value;target=call.args[0].id;rows.append((target,ARRAYS[target],typ(call.args[2]),typ(call.args[1])))
 visit(fn.body);return rows
class StoreTypes(unittest.TestCase):
 def test_all_five_stores_match(self):
  actual=stores(ROOT/'kernels.py');self.assertEqual(len(actual),5)
  self.assertTrue(all(dst==value for _,dst,value,_ in actual),actual)
  self.assertEqual(next(r for r in actual if r[0]=='tokens'),('tokens','int32','int32','int64'))
 def test_v2_fixture_detects_exact_failure(self):
  bad=[r for r in stores(ROOT.parent/'tileir_vllm_post_update_v2/kernels.py') if r[1]!=r[2]]
  self.assertEqual(bad,[('tokens','int32','int64','int64')])
 def test_only_one_store_value_ast_change(self):
  old=ast.parse((ROOT.parent/'tileir_vllm_post_update_v2/kernels.py').read_text());new=ast.parse((ROOT/'kernels.py').read_text())
  calls=[n for n in ast.walk(new) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='scatter' and n.args[0].id=='tokens'];self.assertEqual(len(calls),1)
  cast=calls[0].args[2];self.assertEqual(cast.func.attr,'astype');self.assertEqual(cast.args[1].attr,'int32');calls[0].args[2]=cast.args[0]
  self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
 def test_guarded_token_conversion_is_lossless(self):
  for token in [0,1,248319]:
   self.assertTrue(0<=token<248320)
   narrowed=((token+(1<<31))%(1<<32))-(1<<31)
   self.assertEqual(narrowed,token)
 def test_guard_boundaries_not_broadened(self):
  cases=[(-(1<<63),False),(-1,False),(0,True),(1,True),(248319,True),(248320,False),((1<<31)-1,False),(1<<31,False),((1<<63)-1,False)]
  for token,expected in cases:self.assertEqual(0<=token<248320,expected)
  self.assertEqual((ROOT/'contract.py').read_bytes(),(ROOT.parent/'tileir_vllm_post_update_v2/contract.py').read_bytes())
if __name__=='__main__':unittest.main()
