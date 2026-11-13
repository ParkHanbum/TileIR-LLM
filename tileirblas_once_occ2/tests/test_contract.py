import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from tileirblas_once_occ2 import api
from tileirblas_once_occ2 import _enqueue as backend
ROOT=Path(__file__).resolve().parents[1]
def plan(m,n,k):return SimpleNamespace(m=m,n=n,k=k)
class Once(unittest.TestCase):
    def setUp(self):api.reset_stats()
    def test_one_validation_enqueue(self):
        a,w,y=object(),object(),object();p=plan(32,2560,4096)
        with patch.object(api,'validate_gemm',return_value=p) as validate,patch.object(api,'_enqueue',return_value=y) as enqueue:
            self.assertIs(api.linear(a,w,out=y),y)
            validate.assert_called_once_with(a,w,y,transa=False,transb=True,alpha=1.0,beta=0.0,C=None)
            enqueue.assert_called_once_with(a,w,y,p,'decode64k64')
        self.assertEqual(api.stats()['calls'],1);self.assertEqual(api.stats()['launches'],1)
    def test_selection(self):
        cases=[((32,2560,4096),'decode64k64'),((32,2560,9216),'decode64k64'),((1696,8192,2560),'prefill128'),((1696,32,2560),'blocked'),((32,8192,2560),'blocked'),((3,5,33),'residue'),((3,5,0),'blocked'),((0,5,33),'empty'),((3,0,32),'empty')]
        for dims,path in cases:self.assertEqual(api.selected_path(*dims),path)
    def test_launch_descriptors(self):
        cases=[((32,2560,4096),'decode64k64','tileirblas_decode.kernels','decode_kernel',(1,40,1),(4096,64,64)),((3,5,33),'residue','tileirblas_blocked.fallback','nt_kernel',(1,1,1),(3,5,33)),((1696,32,2560),'blocked','tileirblas_blocked.kernels','blocked_kernel',(53,1,1),(1696,32,2560,32,32)),((32,2560,0),'blocked','tileirblas_blocked.kernels','blocked_kernel',(1,20,1),(32,2560,0,32,128)),((100,100,64),'blocked','tileirblas_blocked.kernels','blocked_kernel',(1,1,1),(100,100,64,256,128))]
        for dims,path,module,name,grid,tail in cases:
            a,w,y,kernel=object(),object(),object(),object()
            with patch.object(backend,'_resolve_kernel',return_value=kernel) as resolve,patch.object(backend,'_launch_kernel') as launch,patch.object(backend,'_allocate') as alloc:
                self.assertIs(backend.enqueue(a,w,y,plan(*dims),path),y)
                resolve.assert_called_once_with(module,name);launch.assert_called_once_with(a,kernel,grid,(a,w,y)+tail);alloc.assert_not_called()
    def test_allocation_empty(self):
        a,w,y=object(),object(),object();p=plan(0,5,32)
        with patch.object(backend,'_allocate',return_value=y) as alloc,patch.object(backend,'_resolve_kernel') as resolve,patch.object(backend,'_launch_kernel') as launch:
            self.assertIs(backend.enqueue(a,w,None,p,'empty'),y);alloc.assert_called_once_with(a,p);resolve.assert_not_called();launch.assert_not_called()
    def test_actual_rejections(self):
        a=torch.empty((3,7),dtype=torch.bfloat16);w=torch.empty((5,7),dtype=torch.bfloat16)
        for kw in [{},{'alpha':0},{'beta':1},{'transa':True},{'transb':False},{'C':a}]:
            with patch.object(api,'_enqueue') as enqueue:
                with self.assertRaises((ValueError,NotImplementedError)):api.gemm(a,w,**kw)
                enqueue.assert_not_called()
        self.assertEqual(api.stats()['calls'],0)
    def test_shared_validator_contract(self):
        a=torch.empty((3,7),dtype=torch.bfloat16);w=torch.empty((7,7),dtype=torch.bfloat16)
        for aa,ww,out in [(a,w,a),(a.float(),w,None),(a[:,::2],w,None),(a.clone().requires_grad_(),w,None)]:
            with self.assertRaises(ValueError):api.validate_gemm(aa,ww,out,require_cuda=False)
    def test_failures_and_empty_counters(self):
        with patch.object(api,'validate_gemm',return_value=plan(32,2560,4096)),patch.object(api,'_enqueue',side_effect=RuntimeError('allocation or enqueue')):
            with self.assertRaises(RuntimeError):api.gemm_nt(None,None)
        with patch.object(api,'validate_gemm',return_value=plan(0,5,32)),patch.object(api,'_enqueue',return_value=None):api.gemm_nt(None,None)
        s=api.stats();self.assertEqual((s['calls'],s['failures'],s['empty_returns'],s['launches']),(2,1,1,0))
    def test_dependency_pins(self):
        for name,sha in json.loads((ROOT/'FROZEN-PINS.json').read_text()).items():self.assertEqual(hashlib.sha256((ROOT.parent/name).read_bytes()).hexdigest(),sha,name)
if __name__=='__main__':unittest.main()
