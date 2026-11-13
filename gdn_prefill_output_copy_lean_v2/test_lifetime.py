import unittest,types,sys
from unittest.mock import patch
from .lifetime import Copy,OriginalCopy
class T:
 device='cuda:0'
 def __init__(self,events,label,fail=False):self.events=events;self.label=label;self.fail=fail
 def record_stream(self,s):
  self.events.append(self.label)
  if self.fail:raise RuntimeError('record failure')
class Tests(unittest.TestCase):
 def run_case(self,fail=False,changed=False):
  events=[];stream=types.SimpleNamespace(cuda_stream=7);t=types.ModuleType('torch');t.cuda=types.SimpleNamespace(current_stream=lambda *_:stream)
  d,s,v,w=[T(events,n,fail and n=='src')for n in('dst','src','sv','dv')];c=Copy()
  def launch(self,d,s):
   events.append('launch');self.pending.append((d,s,v,w,types.SimpleNamespace(cuda_stream=7)));self.calls+=1
   if changed:stream.cuda_stream=8
   return d
  with patch.dict(sys.modules,{'torch':t}),patch.object(OriginalCopy,'__call__',launch):
   if fail or changed:
    with self.assertRaises(RuntimeError):c(d,s)
    self.assertTrue(c.failed);self.assertEqual(len(c.pending),1);self.assertFalse(c.recorded)
   else:self.assertIs(c(d,s),d);self.assertFalse(c.pending);self.assertTrue(c.recorded);self.assertEqual(events,['launch','dst','src','sv','dv'])
 def test_success(self):self.run_case()
 def test_record_failure_retains(self):self.run_case(fail=True)
 def test_stream_change_retains(self):self.run_case(changed=True)
if __name__=='__main__':unittest.main()
