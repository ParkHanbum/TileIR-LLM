import ast,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
from public_i64_range_guard_v1.api import all_in_range_i64
from public_i64_range_guard_v1.cases import cases
from public_i64_range_guard_v1.contract import reference,MIN,MAX
from .run import result_gate
class FakeTensor:
 def __init__(self,**kw):self.__dict__.update(dict(device=types.SimpleNamespace(type='cuda'),layout='strided',requires_grad=False,dtype='torch.int64',shape=(32,),strides=(1,),offset=0,size=256),**kw)
 def stride(self):return self.strides
 def storage_offset(self):return self.offset
 def untyped_storage(self):return types.SimpleNamespace(nbytes=lambda:self.size)
 def is_neg(self):return False
 def is_conj(self):return False
class Tests(unittest.TestCase):
 def test_prelaunch_hold(self):
  torch=types.SimpleNamespace(Tensor=FakeTensor,strided='strided')
  import builtins
  original=builtins.__import__;seen=[]
  def importing(name,*a,**k):
   if name.startswith('cuda'):seen.append(name);raise AssertionError('Tile imported before unsupported rejection')
   return original(name,*a,**k)
  with patch.dict(sys.modules,{'torch':torch}),patch('builtins.__import__',side_effect=importing):
   for kw,lo,hi in [({'strides':(2,),'size':512},0,32),({'dtype':'torch.int32'},0,32),({'offset':1},0,32),({},MIN-1,0),({},0,MAX+1),({'shape':(1025,),'size':8200},0,32),({'requires_grad':True},0,32)]:
    with self.assertRaises(ValueError):all_in_range_i64(FakeTensor(**kw),lo,hi)
  self.assertEqual(seen,[])
 def test_empty_no_launch(self):
  with patch.dict(sys.modules,{'torch':types.SimpleNamespace(Tensor=FakeTensor,strided='strided')}):self.assertIs(all_in_range_i64(FakeTensor(shape=(0,),size=0),1,0),True)
 def test_cases(self):
  self.assertEqual(len(cases()),16);self.assertTrue(any(s['shape']==[32,6] for s in cases()));self.assertTrue(any(s['shape']==[]for s in cases()))
 def receipt(self):
  rows=[]
  for stream in range(2):
   for i,s in enumerate(cases()):
    expected=reference(s['values'],s['lower'],s['upper']);rows.append(dict(stream=stream,case=i,spec=s,stage='verified',passed=True,native=expected,candidate=expected,readonly_checks=2,stream_unchanged=True))
  return dict(complete=True,records=rows,final_synchronize=True,current_stream_restored=True,model_calls=0,timing=False)
 def test_gate(self):self.assertTrue(result_gate(self.receipt())['raw_pass'])
 def test_corrupt_gate(self):
  for key,value in [('stage','candidate_attempt'),('candidate',1),('readonly_checks',1),('stream_unchanged',False)]:
   r=self.receipt();r['records'][0][key]=value;self.assertFalse(result_gate(r)['raw_pass'])
 def test_ast(self):
  root=Path(__file__).resolve().parent
  for p in list(root.glob('*.py'))+list((root.parent/'public_i64_range_guard_v1').glob('*.py')):ast.parse(p.read_text(),filename=str(p))
