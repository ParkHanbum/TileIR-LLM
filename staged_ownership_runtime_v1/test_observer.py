import unittest
from types import SimpleNamespace as NS
from .observer import OwnershipObserver
class Storage:
    def __init__(self, p, n): self.p, self.n = p, n
    def data_ptr(self): return self.p
    def nbytes(self): return self.n
class Tensor:
    dtype, device = 'torch.int32', 'cuda:0'
    def __init__(self, p=1024, offset=0):
        self.shape, self.offset, self.s = (2, 4), offset, Storage(p, 64)
    def stride(self): return (4, 1)
    def storage_offset(self): return self.offset
    def element_size(self): return 4
    def untyped_storage(self): return self.s
    def data_ptr(self): return self.s.p + self.offset * 4


class Tests(unittest.TestCase):
    def make(self, compare=lambda *a:True, sync=lambda:None):
        class Staged:
            def __init__(self,p=1024): self.gpu=Tensor(p); self._staged_write_indices=[]; self.r=object(); self.n=0
            def apply_write(self): self.n+=1; return self.r
        class Fused:
            def apply(self,tensors,output_ptrs,output_strides): self.n=getattr(self,'n',0)+1; return self
        class Table:
            def __init__(self,owners): self.block_tables=owners; self.init_block_table_layout_tensors()
            def init_block_table_layout_tensors(self):
                self.block_table_ptrs=Tensor(4096); self.block_table_ptrs.dtype='torch.uint64'
                self.block_table_strides=Tensor(8192); self.block_table_strides.dtype='torch.int64'
                for t in (self.block_table_ptrs,self.block_table_strides):
                    t.shape=(len(self.block_tables),); t.stride=lambda:(1,)
        phase=['warmup']
        obs=OwnershipObserver(NS(StagedWriteTensor=Staged,FusedStagedWriter=Fused),NS(BlockTables=Table),phase=lambda:phase[0],compare_entries=compare,synchronize=sync)
        return Staged,Fused,Table,phase,obs
    def test_normal(self):
        S,F,T,phase,o=self.make(); original=S.apply_write
        with o:
            a,b=S(),S(2048); table=T([a,b]); f=F()
            self.assertIs(a.apply_write(),a.r)
            phase[0]='request'; a._staged_write_indices=[0]
            self.assertIs(f.apply([a,b],table.block_table_ptrs,table.block_table_strides),f)
        self.assertIs(S.apply_write,original); self.assertTrue(o.released)
        r=o.safe_receipt(); self.assertEqual(r['counts']['request.fused.native_returns'],1)
        self.assertIsNone(r['actual_write_gpu_count'])
    def test_before_native_and_latch(self):
        S,F,T,p,o=self.make()
        with o:
            a=S(); a.gpu=Tensor(8192)
            with self.assertRaises(RuntimeError): a.apply_write()
            with self.assertRaises(RuntimeError): a.apply_write()
            self.assertEqual(a.n,0); self.assertEqual(o.failure['stage'],'before_native')
    def test_after_native(self):
        S,F,T,p,o=self.make(compare=lambda *a:False)
        with o:
            a=S()
            with self.assertRaises(RuntimeError): T([a])
            self.assertEqual(o.failure['stage'],'after_native')
            self.assertEqual(o.counts['warmup.layout_setup.native_returns'],1)
    def test_sync_failure_retains_refs(self):
        def bad(): raise RuntimeError('sync')
        S,F,T,p,o=self.make(sync=bad); original=S.apply_write
        with self.assertRaises(RuntimeError):
            with o: a=S()
        self.assertIs(S.apply_write,original); self.assertEqual(len(o.views),1)
        self.assertFalse(o.released)
    def test_native_uncertain_no_retry(self):
        S,F,T,p,o=self.make()
        def bad(self): self.n+=1; raise ValueError('native')
        S.apply_write=bad
        with o:
            a=S()
            with self.assertRaises(RuntimeError): a.apply_write()
            with self.assertRaises(RuntimeError): a.apply_write()
            self.assertEqual(a.n,1); self.assertEqual(o.failure['stage'],'native_uncertain')
    def test_foreign_restore_retains_owner_and_other_bindings_restore(self):
        S,F,T,p,o=self.make();old=F.apply;o.__enter__();a=S()
        foreign=lambda self:None;S.apply_write=foreign
        with self.assertRaises(ValueError):o.close()
        self.assertIs(S.apply_write,foreign);self.assertIs(F.apply,old)
        self.assertEqual(len(o.views),1);self.assertFalse(o.released)
    def test_stream_drift_latches_after_native(self):
        S,F,T,p,o=self.make();stream=[object()];o.current_stream=lambda:stream[0]
        old=S.apply_write
        def change(owner):
            result=old(owner);stream[0]=object();return result
        S.apply_write=change
        with o:
            a=S()
            with self.assertRaises(RuntimeError):a.apply_write()
            self.assertEqual(a.n,1);self.assertEqual(o.failure['stage'],'after_native')
            with self.assertRaises(RuntimeError):a.apply_write()
            self.assertEqual(a.n,1)
if __name__=='__main__': unittest.main()
