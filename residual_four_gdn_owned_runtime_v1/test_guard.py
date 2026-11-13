import unittest
from types import SimpleNamespace as NS
from .guard import connect
class Tests(unittest.TestCase):
    def setup(self,fn=None):
        calls=[];c=NS(failed=None,active=True)
        def latch(label,error):
            if c.failed is None:c.failed=label
        c.latch=latch
        marker=object();space={'helper':marker}
        f=NS(controller=c,failed=False,active=True,owned_checks=[(space,'helper',marker)])
        def target(*args,**kw):calls.append(args);return marker
        o=NS(failed=False,zeros=fn or target,contiguous_b=target,contiguous_a=target)
        connect(o,c,f);return o,c,f,calls,space,marker
    def test_identity_once(self):
        o,c,f,calls,space,marker=self.setup()
        self.assertIs(o.zeros(1),marker);self.assertEqual(calls,[(1,)])
    def test_existing_four_failure_blocks_before_target(self):
        o,c,f,calls,*_=self.setup();f.failed=True
        with self.assertRaises(RuntimeError):o.contiguous_b(1)
        self.assertEqual(calls,[]);self.assertTrue(o.failed);self.assertIsNotNone(c.failed)
    def test_owned_global_mutation_latches(self):
        o,c,f,calls,space,_=self.setup();space['helper']=object()
        with self.assertRaises(RuntimeError):o.contiguous_a(1)
        self.assertEqual(calls,[]);self.assertTrue(f.failed)
    def test_gdn_error_latches_all_no_retry(self):
        calls=[]
        def bad(*a):calls.append(1);raise ValueError('fixture')
        o,c,f,_,_,_=self.setup(bad)
        with self.assertRaises(ValueError):o.zeros(1)
        with self.assertRaises(RuntimeError):o.zeros(1)
        self.assertEqual(calls,[1]);self.assertTrue(o.failed and f.failed);self.assertEqual(c.failed,'gdn_owned/zeros')
    def test_wrong_controller(self):
        o,c,f,*_=self.setup();f.controller=object()
        with self.assertRaises(RuntimeError):connect(o,c,f)
if __name__=='__main__':unittest.main()
