import ast
import types
import unittest
from .shadow import Binding, ShadowOps
from tileir_gdn_owned_copies_v1.transform import transform_forward
class Owner:
    gqa_interleaved_layout=False; enable_fused_gdn_spec_decode=False
    def forward_cuda(self,x): return x
    def __init__(self): self._forward_method=self.forward_cuda
class Tests(unittest.TestCase):
    def test_cached_binding_and_restore(self):
        owners=[Owner() for _ in range(24)]; originals=[o._forward_method for o in owners]
        def candidate(self,x): return x
        b=Binding(owners,Owner.forward_cuda,candidate); b.install()
        marker=object(); self.assertIs(owners[0]._forward_method(marker),marker)
        b.restore()
        self.assertTrue(all(o._forward_method is f for o,f in zip(owners,originals)))
    def test_class_patch_does_not_imply_cached_patch(self):
        owners=[Owner() for _ in range(24)]
        original=Owner.forward_cuda
        try:
            Owner.forward_cuda=lambda self,x:x
            with self.assertRaises(RuntimeError): Binding(owners,Owner.forward_cuda,Owner.forward_cuda).install()
            self.assertIs(owners[0]._forward_method.__func__,original)
        finally: Owner.forward_cuda=original
    def test_wrong_scope_rejected_before_any_patch(self):
        for mode in ('fused','gqa','wrapped','count'):
            owners=[Owner() for _ in range(24)]; first=owners[0]._forward_method
            if mode=='fused': owners[-1].enable_fused_gdn_spec_decode=True
            if mode=='gqa': owners[-1].gqa_interleaved_layout=True
            if mode=='wrapped': owners[-1]._forward_method=lambda x:x
            if mode=='count': owners.pop()
            with self.assertRaises(RuntimeError): Binding(owners,Owner.forward_cuda,Owner.forward_cuda).install()
            self.assertIs(owners[0]._forward_method,first)
    def test_restore_conflict_not_clobbered(self):
        owners=[Owner() for _ in range(24)]; b=Binding(owners,Owner.forward_cuda,Owner.forward_cuda); b.install()
        marker=lambda x:x; owners[0]._forward_method=marker
        with self.assertRaises(RuntimeError): b.restore()
        self.assertIs(owners[0]._forward_method,marker)
    def test_scope_and_order(self):
        with self.assertRaises(ValueError): transform_forward(ast.parse('def forward_cuda():\n return None').body[0])
        s=ShadowOps(None,None)
        with self.assertRaises(RuntimeError): s._begin('b')
        self.assertEqual(s.calls,0)
if __name__=='__main__': unittest.main()
