"""CPU orchestration regressions; numerical equivalence requires later GPU proof."""
import unittest
from types import SimpleNamespace
from .adapter import Controller,SPECS,ORDER
from lean_public_composition_v1.adapter import SPECS as OLD
class TestComposition(unittest.TestCase):
 def test_original_references_and_counts(self):
  self.assertEqual(len(SPECS),25);self.assertEqual(sum(n for _,_,n in SPECS),5888)
  old={k:(c,n)for k,c,n in OLD}
  for k,c,n in SPECS[:18]:
   if k=='sampler_four':continue
   self.assertIs(c,old[k][0]);self.assertEqual(n,old[k][1])
  self.assertEqual(ORDER[-1],'gdn_parent_state');self.assertLess(ORDER.index('post_conv'),ORDER.index('gdn_prefill_core'))
 def test_partial_install_reverse_cleanup_data_identity(self):
  seen=[];token=object()
  class Part:
   def __init__(self,c,k,n):self.key=k;self.dispatch=token
   def install(self,m):
    seen.append('install:'+self.key)
    if self.key=='fail':raise ValueError('partial')
   def close(self):self_test.assertIs(self.dispatch,token);seen.append('close:'+self.key);return dict(restored=True)
  self_test=self;c=Controller(object(),(('first',Part,1),('fail',Part,1)))
  with self.assertRaises(ValueError):c.install()
  self.assertEqual(seen,['install:first','install:fail','close:fail','close:first']);self.assertEqual(c.parts,[])
 def test_method_specific_no_blanket_entry_guards(self):
  from pathlib import Path
  s=(Path(__file__).parent/'adapter.py').read_text()
  self.assertNotIn('EntryGuards',s);self.assertNotIn('setattr(obj',s)
if __name__=='__main__':unittest.main()
