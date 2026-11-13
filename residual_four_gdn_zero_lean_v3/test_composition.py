import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from .zero_hooks import Hooks
from .cleanup import finish_all
class Tests(unittest.TestCase):
    def make(self):
        calls=[]
        class Z:
            def zero_block_ids(self,ids):calls.append(tuple(ids))
            def warmup(self,num):return self.zero_block_ids([0])
        class R:
            def _init_kv_zero_meta(self):pass
        u=NS(KVBlockZeroer=Z,AttentionSpec=object,torch=NS(Tensor=object))
        mr=NS(GPUModelRunner=R,KVBlockZeroer=Z)
        h=Hooks().install(mr,u,validator=lambda *a:None);z=Z()
        old=NS(strong_owners=[object()]);runner=NS(kv_block_zeroer=z)
        h.diag.setup_returned=1;h.diag.retained=[runner,z];h.diag.registry=old;h.diag.zeroer=z
        return h,z,calls,old
    def test_native_phase_and_fresh_registry(self):
        h,z,calls,old=self.make();z.warmup(1)
        h.begin_native_reference();z.zero_block_ids(list(range(32)))
        fresh=NS(strong_owners=[object()],segments=dict.fromkeys(range(8)),check=lambda:None)
        with patch('zero_kv_typed_lean_v1.registry.collect_runner',return_value=fresh) as collect:h.begin_request()
        self.assertIs(h.diag.registry,fresh);self.assertIs(h.previous_registries[0],old)
        self.assertEqual(calls,[(0,),tuple(range(32))]);self.assertEqual(collect.call_count,1)
        h.controller=NS(failed=None,active=True)
        got=[];h.diag.zero=lambda native,owner,ids:got.append(tuple(ids))
        z.zero_block_ids(list(range(100,132)))
        self.assertEqual(got,[tuple(range(100,132))]);self.assertEqual(len(calls),2)
        h.close();h.release(True)
    def test_repeated_native_no_retry(self):
        h,z,calls,_=self.make();h.begin_native_reference();z.zero_block_ids(list(range(32)))
        with self.assertRaises(Exception):z.zero_block_ids(list(range(32)))
        self.assertEqual(len(calls),1);self.assertIsNotNone(h.failed);h.close()
    def test_cleanup_order_and_retention(self):
        log=[];hooks=NS(close=lambda:log.append('zero_restore'),release=lambda **kw:log.append('release'),receipt=lambda:{})
        result={}
        def final(propagating):log.append('four_base');result.update(synchronized_before_close=True,synchronized_after_close=True,engine_shutdown=True)
        finish_all(result,lambda:None,hooks,final,propagating=False)
        self.assertEqual(log,['zero_restore','four_base','release'])
        log.clear();result={}
        finish_all(result,lambda:None,hooks,lambda p:log.append('four_failure'),propagating=True)
        self.assertEqual(log,['zero_restore','four_failure'])
if __name__=='__main__':unittest.main()
