import struct
import unittest
from types import SimpleNamespace as NS
from .shadow import ShadowOps,cpu_copy_oracle
class T:
    dtype='bf16';device=NS(type='cuda');requires_grad=False
    def __init__(self,data,shape,ptr=1000,offset=0,stride=None):
        self.data=data;self.shape=shape;self.ptr=ptr;self.offset=offset;self.st=stride or (64,1)
    def stride(self):return self.st
    def storage_offset(self):return self.offset
    def is_conj(self):return False
    def is_neg(self):return False
    def untyped_storage(self):return NS(data_ptr=lambda:self.ptr,nbytes=lambda:len(self.data))
    def as_strided(self,shape,strides,storage_offset=0):return T(self.data,shape,self.ptr,storage_offset,strides)
    def view(self,*args):return self
    def cpu(self):return self
    def numpy(self):return self
    def tobytes(self,order='C'):return bytes(self.data)
    def numel(self):
        n=1
        for x in self.shape:n*=x
        return n
    def contiguous(self):raise AssertionError('native target forbidden')
class C:
    def __init__(self):self.ptr=10000000;self.outputs=[]
    def out(self,data,shape):
        self.ptr+=2000000;t=T(data,shape,self.ptr);self.outputs.append(t);return t
    def zeros(self,shape,**kwargs):return self.out(bytes(shape[0]*32*128*2),shape)
    def contiguous_b(self,x):return self.out(cpu_copy_oracle(x.data,x.shape[0],0),x.shape)
    def contiguous_a(self,x):return self.out(cpu_copy_oracle(x.data,x.shape[0],32),x.shape)
class Tests(unittest.TestCase):
    def test_oracle_payloads_and_extent(self):
        patterns=(0,0x8000,0x7f80,0xff80,0x7f81,0x7fc5,1,0x8001)
        for m in (32,1696):
            data=struct.pack('<'+str(m*64)+'H',*(patterns[i%len(patterns)] for i in range(m*64)))
            for offset in (0,32):
                expected=b''.join(data[(r*64+offset)*2:(r*64+offset+32)*2] for r in range(m))
                self.assertEqual(cpu_copy_oracle(data,m,offset),expected)
        with self.assertRaises(ValueError):cpu_copy_oracle(b'',32,0)
    def test_returns_candidates_native_zero(self):
        c=C();o=ShadowOps(NS(uint16='uint16',bfloat16='bf16'),c)
        zero=o.zeros((32,32,128),dtype='bf16',device=NS(type='cuda'))
        data=bytes(i%256 for i in range(32*64*2))
        b=o.contiguous_b(T(data,(32,32)))
        a=o.contiguous_a(T(data,(32,32),offset=32))
        self.assertIs(zero,c.outputs[0]);self.assertIs(b,c.outputs[1]);self.assertIs(a,c.outputs[2])
        r=o.receipt();self.assertEqual(r['passed'],3);self.assertEqual(r['native_target_calls'],0)
        self.assertTrue(r['returns_candidate'])
    def test_bad_zero_latches(self):
        c=C();c.zeros=lambda shape,**kw:c.out(b'\x01'+bytes(32*32*128*2-1),shape)
        o=ShadowOps(NS(uint16='uint16'),c)
        with self.assertRaises(RuntimeError):o.zeros((32,32,128),dtype='bf16',device=None)
        self.assertTrue(o.failed);self.assertEqual(o.passed,0)
if __name__=='__main__':unittest.main()
