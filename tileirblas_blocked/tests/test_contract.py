import unittest
import torch
from tileirblas_blocked import validate_gemm, k_tiles, stats, reset_stats, linear

class Contract(unittest.TestCase):
    def tensors(self, m=3, n=5, k=7):
        return torch.empty((m,k),dtype=torch.bfloat16), torch.empty((n,k),dtype=torch.bfloat16)
    def test_basic(self):
        a,w=self.tensors(); p=validate_gemm(a,w,require_cuda=False)
        self.assertEqual((p.m,p.n,p.k),(3,5,7))
    def test_padded(self):
        a=torch.empty_strided((32,2560),(135680,1),dtype=torch.bfloat16)
        w=torch.empty((2,2560),dtype=torch.bfloat16)
        self.assertEqual(validate_gemm(a,w,require_cuda=False).a_row_stride,135680)
    def test_zero(self):
        for shape in [(0,5,7),(3,0,7),(3,5,0),(0,0,0)]:
            a,w=self.tensors(*shape); validate_gemm(a,w,require_cuda=False)
    def test_order(self):
        for k in (0,1,15,16,17,31,32,33,65,2560):
            tiles=k_tiles(k)
            self.assertEqual([i for s,c in tiles for i in range(s,s+c)],list(range(k)))
            if k: self.assertEqual(tiles[0][1],(k-1)%32+1)
    def test_reject(self):
        a,w=self.tensors()
        for kw in [dict(alpha=0),dict(beta=1),dict(transa=True),dict(transb=False),dict(C=a),dict(alpha=float('nan'))]:
            with self.assertRaises(NotImplementedError): validate_gemm(a,w,require_cuda=False,**kw)
        with self.assertRaises(ValueError): linear(a,w)
        with self.assertRaises(ValueError): validate_gemm(a.float(),w,require_cuda=False)
        with self.assertRaises(ValueError): validate_gemm(a[:,::2],w,require_cuda=False)
    def test_output_and_alias(self):
        a,w=self.tensors(3,7,7)
        with self.assertRaises(ValueError): validate_gemm(a,w,out=a,require_cuda=False)
        out=torch.full((3,7),9,dtype=torch.bfloat16); before=out.clone()
        validate_gemm(a,w,out=out,require_cuda=False)
        self.assertTrue(torch.equal(before,out))
        with self.assertRaises(ValueError): validate_gemm(a,w,out=torch.empty_strided((3,7),(9,1),dtype=torch.bfloat16),require_cuda=False)
    def test_counters(self):
        reset_stats(); self.assertEqual(stats(),dict(calls=0,launches=0,empty_returns=0))

class SourceContract(unittest.TestCase):
    def test_all_qwen_metadata(self):
        shapes=[(1696,8192,2560),(1696,4096,2560),(1696,32,2560),(1696,2560,4096),(1696,9216,2560),(1696,2560,9216),(1696,1024,2560),(32,248320,2560),(32,2560,4096),(32,9216,2560),(32,2560,9216),(32,8192,2560)]
        count=0
        for m,n,k in shapes:
            for stride in ([k,135680] if (m,n,k)==(32,248320,2560) else [k]):
                a=torch.empty_strided((m,k),(stride,1),dtype=torch.bfloat16,device='meta')
                w=torch.empty((n,k),dtype=torch.bfloat16,device='meta')
                plan=validate_gemm(a,w,require_cuda=False)
                self.assertEqual((plan.m,plan.n,plan.k,plan.a_row_stride),(m,n,k,stride)); count+=1
        self.assertEqual(count,13)
    def test_index_masks_and_padding(self):
        m,n,k=3,5,33; sa,sw,sy=40,48,8
        a=[1.0]*(m*sa); w=[2.0]*(n*sw); y=[999.0]*(m*sy)
        for i in range(4):
            for j in range(8):
                accum=0.0
                for start,valid in k_tiles(k):
                    for group in range(2):
                        for lane in range(16):
                            kk=start+group*16+lane
                            if i<m and j<n and group*16+lane<valid:
                                accum+=a[i*sa+kk]*w[j*sw+kk]
                if i<m and j<n: y[i*sy+j]=accum
        for i in range(m):
            self.assertEqual(y[i*sy:i*sy+n],[66.0]*n)
            self.assertEqual(y[i*sy+n:(i+1)*sy],[999.0]*(sy-n))
    def test_bf16_rn_expected(self):
        # CPU oracle of documented RN conversion, not a GPU execution claim.
        vals=torch.tensor([1+2**-8,1+3*2**-8,0.0,float('inf'),float('-inf')],dtype=torch.float32)
        got=vals.to(torch.bfloat16).float()
        self.assertEqual(got[:3].tolist(),[1.0,1+2**-6,0.0])
        self.assertTrue(torch.isinf(got[3:]).all())
    def test_package_import_lazy(self):
        import subprocess,sys
        p=subprocess.run([sys.executable,'-c','import sys,tileirblas_blocked; assert "torch" not in sys.modules; assert "cuda.tile" not in sys.modules'],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)

if __name__=='__main__': unittest.main()
