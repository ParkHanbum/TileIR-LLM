import unittest,ast,hashlib
from pathlib import Path
from types import SimpleNamespace as N
from .proof import verify_and_zero
from unittest.mock import patch
from .registry import Hold
class Tests(unittest.TestCase):
 def setUp(self):
  self.scope=patch(__package__+'.proof.validate_scope',lambda *args:None);self.scope.start();self.addCleanup(self.scope.stop)
 def setup_model(self):
  data=bytearray(range(64));a=N(size=64);s=N(allocation=a,offset=8,stride=16,page=8,blocks=3)
  r=N(check=lambda:None,allocations={0:a},segments={0:s})
  def read(a,l,h,target):return all(x==0 for x in data[l:h])if target else hashlib.sha256(data[l:h]).digest()
  def zero(a,iv,ledger):
   for l,h in iv:ledger['attempted']+=1;data[l:h]=bytes(h-l);ledger['returned']+=1
  return r,data,read,zero
 def test_bounded_full_owner(self):
  r,d,read,zero=self.setup_model();p={};verify_and_zero(r,[0,2],read_chunk=read,zero=zero,synchronize=lambda:None,record=p)
  self.assertTrue(p['target_all_zero'] and p['outside_hash_preserved']);self.assertEqual(p['target_bytes'],16);self.assertEqual(p['zero_returned'],2)
 def test_noop_and_outside_corruption_fail(self):
  for corrupt in(False,True):
   r,d,read,zero=self.setup_model()
   def bad(a,iv,ledger):
    if corrupt:zero(a,iv,ledger);d[0]^=1
    else:ledger.update(attempted=len(iv),returned=len(iv))
   with self.assertRaises(Hold):verify_and_zero(r,[0,2],read_chunk=read,zero=bad,synchronize=lambda:None,record={})
 def test_exception_retains_attempts(self):
  r,d,read,zero=self.setup_model();p={}
  def bad(a,iv,ledger):ledger['attempted']=1;raise RuntimeError('private')
  with self.assertRaises(RuntimeError):verify_and_zero(r,[0],read_chunk=read,zero=bad,synchronize=lambda:None,record=p)
  self.assertEqual((p['zero_attempted'],p['zero_returned']),(1,0))
 def test_source_only(self):
  root=Path(__file__).parent;self.assertFalse((root/'cache_v1').exists());self.assertEqual(list(root.glob('CACHE-PRESEED*')),[])
  for p in root.glob('*.py'):ast.parse(p.read_text())

class SafetyRegression(unittest.TestCase):
 def test_bad_scope_and_budget_no_payload(self):
  from . import proof
  a=N(size=10*1024**3);reg=N(check=lambda:None,allocations={0:a},segments={i:None for i in range(8)})
  counters=dict(read=0,zero=0,sync=0)
  def forbidden(*args,**kwargs):raise AssertionError('payload access before guard')
  plans=[(a,[('target',0,553648128),('outside',553648128,a.size)])]
  with patch.object(proof,'from_registry',return_value=plans):
   with self.assertRaisesRegex(Hold,'proof_io_budget'):proof.verify_and_zero(reg,list(range(32)),read_chunk=forbidden,zero=forbidden,synchronize=forbidden,record={})
   with self.assertRaisesRegex(Hold,'actual_geometry_scope'):proof.verify_and_zero(reg,[0]*32,read_chunk=forbidden,zero=forbidden,synchronize=forbidden,record={})
  for target,allocs,segs,ids in [(1,1,8,list(range(32))),(553648128,2,8,list(range(32))),(553648128,1,7,list(range(32)))]:
   with self.assertRaises(Hold):proof.validate_scope(N(allocations={i:None for i in range(allocs)},segments={i:None for i in range(segs)}),ids,target)
 def test_shutdown_before_release_and_failure_retains(self):
  from .cleanup import teardown
  for fail in(False,True):
   events=[]
   def shutdown():
    events.append('shutdown')
    if fail:raise RuntimeError('fail')
   h=N(close=lambda:events.append('restore'),release=lambda **kw:events.append('release'));e=N(engine_core=N(shutdown=shutdown));r={'cleanup_errors':[]}
   teardown(h,None,e,lambda:events.append('sync'),r)
   self.assertEqual(events,['restore','sync','shutdown']+([]if fail else ['release']))
   self.assertEqual(r.get('owners_retained',False),fail)
