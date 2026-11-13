import unittest,tempfile,json
from pathlib import Path
from runner_scalar32_qualification_v3.test_cpu import owner,raw,candidate
from runner_scalar32_qualification_v3.qualifier import ScalarQualifier
from runner_scalar32_qualification_v3.ledger import DurableLedger
class Tests(unittest.TestCase):
 def shadow(self,q,x):
  q.binding.install();q.start('shadow')
  for i in range(32):x.add_request(i,None)
  q.finish()
 def test_changed_order_and_preimage(self):
  x=owner();q=ScalarQualifier(x,candidate,raw);self.shadow(q,x)
  x.num_accepted_tokens_gpu.v=list(range(-100,-68));q.start('replace')
  for i in reversed(range(32)):x.add_request(i,None)
  self.assertEqual(q.finish()['calls'],32)
  self.assertEqual(x.num_accepted_tokens_gpu.v,[1]*32)
  self.assertFalse(any(e['stage']=='native_original_intent' for e in q.ledger if e['phase']=='replace'));q.close()
 def test_durable_candidate_failure(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'ledger';x=owner();q=ScalarQualifier(x,candidate,raw,DurableLedger(p));self.shadow(q,x);q.start('replace')
   x.num_accepted_tokens_gpu.v=list(range(32))
   def bad(v,i):v.v[i]=1;raise RuntimeError('candidate after mutation')
   q.candidate=bad
   with self.assertRaises(RuntimeError):x.add_request(0,None)
   events=[json.loads(s) for s in p.read_text().splitlines() if s];events=[e for e in events if e['phase']=='replace']
   self.assertEqual(events[-1]['stage'],'failure')
   self.assertEqual(sum(e['stage']=='candidate_intent' for e in events),1)
   self.assertEqual(sum(e['stage']=='candidate_return' for e in events),0);q.close()
 def test_out_of_slot_corruption_detected(self):
  x=owner();q=ScalarQualifier(x,candidate,raw);self.shadow(q,x);q.start('replace')
  def bad(v,i):v.v[i]=1;v.v[31]=999
  q.candidate=bad
  with self.assertRaises(ValueError):x.add_request(0,None)
  self.assertEqual(q.ledger[-1]['stage'],'failure');q.close()
 def test_duplicate_before_candidate(self):
  x=owner();q=ScalarQualifier(x,candidate,raw);q.binding.install();q.start('shadow');x.add_request(0,None)
  with self.assertRaises(ValueError):x.add_request(0,None)
  self.assertEqual(sum(e['stage']=='candidate_intent'for e in q.ledger),1);q.close()
 def test_record_failure_blocks_candidate(self):
  x=owner();calls=[]
  def fail(e):raise OSError('disk')
  q=ScalarQualifier(x,lambda *a:calls.append(1),raw,fail);q.binding.install();q.start('shadow')
  with self.assertRaises(OSError):x.add_request(0,None)
  self.assertFalse(calls);self.assertTrue(q.failed);q.close()
if __name__=='__main__':unittest.main()
