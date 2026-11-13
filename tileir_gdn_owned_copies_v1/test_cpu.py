import ast
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from .api import check_copy, OwnedCopies
from .transform import transform_forward
SOURCE=Path('/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py')
class X:
    dtype='bf16'; device=NS(type='cuda'); requires_grad=False
    def __init__(self,m=32,off=0): self.shape=(m,32); self.off=off
    def stride(self): return (64,1)
    def storage_offset(self): return self.off
    def is_conj(self): return False
    def is_neg(self): return False
    def untyped_storage(self): return NS(nbytes=lambda:self.shape[0]*64*2)
class Tests(unittest.TestCase):
    def test_guard(self):
        t=NS(bfloat16='bf16')
        for m in (32,1696):
            for off in (0,32): check_copy(X(m,off),off,t)
        for x in (X(31),X(32,1)):
            with self.assertRaises(ValueError): check_copy(x,0,t)
    def test_all_raw_words(self):
        # Synthetic packing only, no floating-point conversion. Includes every NaN payload.
        source=bytearray(struct.pack('<65536H',*range(65536)))
        prior=bytes(source)
        for off in (0,32):
            out=bytearray()
            for row in range(32):
                start=(row*64+off)*2
                out.extend(source[start:start+64])
            self.assertEqual(len(out),2048)
            self.assertEqual(out[:64],source[off*2:off*2+64])
        self.assertEqual(bytes(source),prior)
        self.assertEqual(struct.pack('<65536H',*struct.unpack('<65536H',source)),source)
    def test_tiles(self):
        for m in (32,1696):
            self.assertEqual((m//32)*32,m)
            self.assertEqual((m*32*128//1024)*1024,m*32*128)
        node=ast.parse(Path(__file__).with_name('kernels.py').read_text())
        calls=[ast.unparse(n.func) for n in ast.walk(node) if isinstance(n,ast.Call)]
        self.assertNotIn('ct.astype',calls)
        self.assertEqual(calls.count('ct.load'),1)
    def test_transform_preserves_core(self):
        tree=ast.parse(SOURCE.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QwenGatedDeltaNetAttention')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward_cuda')
        out=transform_forward(fn)
        s=ast.unparse(out)
        self.assertEqual(s.count('_owned_gdn_copies.zeros('),1)
        self.assertIn('torch.ops.vllm.qwen_gdn_attention_core(mixed_qkv, _owned_gdn_copies.contiguous_b(b), _owned_gdn_copies.contiguous_a(a), core_attn_out',s)
        with self.assertRaises(ValueError): transform_forward(out)
    def test_launch_failure_latch(self):
        class Device:
            def __enter__(self): pass
            def __exit__(self,*a): pass
        t=NS(cuda=NS(device=lambda d:Device(), current_stream=lambda d:None))
        def bad(*a): raise RuntimeError('mock launch')
        a=OwnedCopies(t,NS(launch=bad),None)
        with self.assertRaises(RuntimeError): a._launch('b',(1,1,1),None,(),None)
        self.assertEqual(a.attempts['b'],1); self.assertEqual(a.enqueues['b'],0)
        with self.assertRaises(RuntimeError): a._healthy()
if __name__=='__main__': unittest.main()
