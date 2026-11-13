import unittest,copy,types
from .evidence import check_rows,check_ledger,check_stream,SCOPE
from tileir_staged_typed_v2.api import apply
class Tests(unittest.TestCase):
 def fixture(self):
  rows=[];counts={}
  for label,total,n in [('staged',123,5),('fused',17,1)]:
   counts['request.'+label+'.nonempty']=n;counts['request.'+label+'.empty']=total-n
   for i in range(total):
    flags=[i<n]if label=='staged'else[i<n,False]
    rows.append(dict(label=label,nonempty=any(flags),typed_nonempty=flags,candidate_stream_unchanged=True,launch_ledger=[dict(nonempty=a,launch_attempts=int(a),launch_returns=int(a),scope=SCOPE)for a in flags]))
  return rows,counts
 def test_good(self):self.assertTrue(check_rows(*self.fixture()))
 def test_wrong_distribution_same_total(self):
  r,c=self.fixture();r[0]['nonempty']=False;r[-1]['nonempty']=True
  with self.assertRaises(AssertionError):check_rows(r,c)
 def test_empty_launch_rejected(self):
  r,c=self.fixture();r[10]['launch_ledger'][0]['launch_attempts']=1
  with self.assertRaises(AssertionError):check_rows(r,c)
 def test_missing_typed_launch_rejected(self):
  r,c=self.fixture();r[123]['launch_ledger'][0]['launch_returns']=0
  with self.assertRaises(AssertionError):check_rows(r,c)
 def test_stream_drift_rejected(self):
  with self.assertRaises(AssertionError):check_stream(object(),object())
  r,c=self.fixture();r[0]['candidate_stream_unchanged']=False
  with self.assertRaises(AssertionError):check_rows(r,c)
 def test_actual_empty_ledger(self):
  t=types.SimpleNamespace(shape=(2,),stride=lambda:(1,),dtype='torch.int32',device=types.SimpleNamespace(type='cuda'),requires_grad=False,storage_offset=lambda:0,numel=lambda:2,element_size=lambda:4,untyped_storage=lambda:types.SimpleNamespace(nbytes=lambda:8));ledger=[]
  self.assertIsNone(apply(t,[],[],[],[],ledger=ledger));self.assertEqual(ledger,[dict(nonempty=False,launch_attempts=0,launch_returns=0,scope=SCOPE)])
 def test_actual_launch_attempt_return_ledger(self):
  import sys
  from unittest.mock import patch
  t=types.SimpleNamespace(shape=(2,),stride=lambda *a:(1,)if not a else 1,dtype='torch.int32',device=types.SimpleNamespace(type='cuda'),requires_grad=False,storage_offset=lambda:0,numel=lambda:2,element_size=lambda:4,untyped_storage=lambda:types.SimpleNamespace(nbytes=lambda:8))
  t.view=lambda *a:t
  for fail in (False,True):
   events=[];ledger=[]
   stream=types.SimpleNamespace(synchronize=lambda:events.append('sync'))
   torch=types.ModuleType('torch');torch.int32='torch.int32';torch.tensor=lambda *a,**k:object();torch.cuda=types.SimpleNamespace(current_stream=lambda *a:stream)
   tile=types.ModuleType('cuda.tile');cuda=types.ModuleType('cuda');cuda.tile=tile
   def launch(*a):
    self.assertEqual(ledger[0]['launch_attempts'],1);self.assertEqual(ledger[0]['launch_returns'],0);events.append('launch')
    if fail:raise RuntimeError('mock compile/launch failed')
   tile.launch=launch;kernels=types.ModuleType('tileir_staged_typed_v2.kernels');kernels.write_segments=object()
   with patch.dict(sys.modules,{'torch':torch,'cuda':cuda,'cuda.tile':tile,'tileir_staged_typed_v2.kernels':kernels}):
    if fail:
     with self.assertRaises(RuntimeError):apply(t,[0],[0],[7],[1],ledger=ledger)
    else:self.assertIsNone(apply(t,[0],[0],[7],[1],ledger=ledger))
   self.assertEqual(ledger[0]['launch_attempts'],1);self.assertEqual(ledger[0]['launch_returns'],int(not fail));self.assertEqual(events,['launch']if fail else['launch','sync'])
