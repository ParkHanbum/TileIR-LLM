"""Process-local diagnostic only; caller supplies imported public modules/callbacks."""
from collections import Counter
from .ownership import ViewLease, require

class OwnershipObserver:
    def __init__(self, buffer_module, block_module, *, phase, compare_entries, synchronize, current_stream=None):
        self.b, self.bt = buffer_module, block_module
        self.phase, self.compare, self.sync = phase, compare_entries, synchronize
        self.counts, self.failure = Counter(), None
        self.views, self.layouts, self.patches = [], [], []
        self.released = False
        self.current_stream=current_stream
        self.stream_refs=[]

    def _view(self, owner):
        found = [v for v in self.views if v.owner is owner]
        require(len(found) == 1, 'unregistered or duplicate owner')
        found[0].check()
        return found[0]

    def _table_check(self, owners, ptrs, strides):
        records = [r for r in self.layouts if r[1] is ptrs and r[2] is strides]
        require(len(records) == 1, 'unregistered table')
        registered = records[0][0]
        require(len(owners) == len(registered) and all(a is b for a,b in zip(owners,registered)),
                'owner order changed')
        vs = [self._view(o) for o in owners]
        require(0 < len(vs) <= 4 and all(v.dtype == 'torch.int32' for v in vs), 'groups')
        require(len({v.device for v in vs}) == 1, 'group device')
        for table, dtype in ((ptrs,'torch.uint64'),(strides,'torch.int64')):
            require(str(table.dtype) == dtype and str(table.device) == vs[0].device
                    and tuple(table.shape) == (len(vs),) and tuple(table.stride()) == (1,)
                    and table.storage_offset() == 0, 'table metadata')
        for i,a in enumerate(vs):
            for b in vs[i+1:]:
                ap,bp = a.storage.data_ptr(),b.storage.data_ptr()
                require(ap+a.nbytes <= bp or bp+b.nbytes <= ap, 'backing overlap')
        require(self.compare(ptrs,strides,tuple(v.tensor for v in vs)) is True, 'entry equality')
        self.counts[self.phase()+'.table_equality_checks']+=1

    def _run(self, label, native, args, kwargs, pre=None, post=None):
        if self.failure is not None:
            self.counts['latched_rejections'] += 1
            raise RuntimeError('ownership observer latched') from None
        phase = self.phase()
        require(phase in ('setup','warmup','request','cleanup'), 'phase')
        key = phase + '.' + label
        self.counts[key+'.calls'] += 1
        stage = 'before_native'
        try:
            require(all(getattr(cls,name)is wrapper for cls,name,original,wrapper in self.patches),'binding drift')
            stream=self.current_stream() if self.current_stream else None
            if stream is not None:self.stream_refs.append(stream)
            if pre: pre(*args, **kwargs)
            if stream is not None:require(self.current_stream()==stream,'pre stream changed')
            self.counts[key+'.native_attempts'] += 1
            stage = 'native_uncertain'
            result = native(*args, **kwargs)
            self.counts[key+'.native_returns'] += 1
            stage = 'after_native'
            if post: post(*args, **kwargs)
            if stream is not None:
                require(self.current_stream()==stream,'native stream changed')
                self.counts[key+'.stream_unchanged']+=1
            return result
        except BaseException:
            self.failure = {'phase':phase,'label':label,'stage':stage}
            self.counts[key+'.failures.'+stage] += 1
            raise RuntimeError('ownership diagnostic failed: '+stage) from None

    def __enter__(self):
        require(not self.patches and not self.released, 'context reuse')
        def staged_setup(owner, *args, **kwargs):
            require(not any(v.owner is owner for v in self.views), 'duplicate setup')
            self.views.append(ViewLease.capture(owner))
        def layout_setup(owner, *args, **kwargs):
            owners = tuple(owner.block_tables)
            for o in owners: self._view(o)
            self.layouts.append((owners,owner.block_table_ptrs,owner.block_table_strides,owner))
            self._table_check(owners,owner.block_table_ptrs,owner.block_table_strides)
        def staged_pre(owner):
            self._view(owner)
            self.counts[self.phase()+'.staged.'+('nonempty' if len(owner._staged_write_indices) else 'empty')] += 1
        def fused_pre(owner,tensors,output_ptrs,output_strides):
            self._table_check(tensors,output_ptrs,output_strides)
            self.counts[self.phase()+'.fused.'+('nonempty' if any(len(t._staged_write_indices) for t in tensors) else 'empty')] += 1
        def staged_post(owner):
            self._view(owner)
            self.counts[self.phase()+'.staged.alias_preserved']+=1
        def fused_post(owner,tensors,output_ptrs,output_strides):
            self._table_check(tensors,output_ptrs,output_strides)
            self.counts[self.phase()+'.fused.alias_preserved']+=1
        targets = [(self.b.StagedWriteTensor,'__init__','staged_setup',None,staged_setup),
                   (self.bt.BlockTables,'init_block_table_layout_tensors','layout_setup',None,layout_setup),
                   (self.b.StagedWriteTensor,'apply_write','staged',staged_pre,staged_post),
                   (self.b.FusedStagedWriter,'apply','fused',fused_pre,fused_post)]
        try:
            for cls,name,label,pre,post in targets:
                original = getattr(cls,name)
                require(not getattr(original,'_staged_ownership_observer',False),'nested observer')
                def wrapper(*args,_native=original,_label=label,_pre=pre,_post=post,**kwargs):
                    return self._run(_label,_native,args,kwargs,_pre,_post)
                wrapper._staged_ownership_observer=True
                self.patches.append((cls,name,original,wrapper))
                setattr(cls,name,wrapper)
        except BaseException:
            self._restore()
            raise
        return self

    def _restore(self):
        failed=[]
        for cls,name,original,wrapper in reversed(self.patches):
            if getattr(cls,name) is not wrapper:
                failed.append((cls,name,original,wrapper));continue
            setattr(cls,name,original)
        self.patches=list(reversed(failed))
        require(not failed,'foreign binding; owners retained')

    def __exit__(self,*exc):
        self._restore()
        try:
            self.sync()
        except BaseException:
            self.counts['cleanup_sync_failed_refs_retained'] += 1
            raise RuntimeError('ownership cleanup sync failed; retain observer refs') from None
        self.views.clear(); self.layouts.clear(); self.stream_refs.clear(); self.released=True
        self.counts['cleanup_sync_passed_refs_released'] += 1
        return False

    def close(self):
        """Restore only; caller must synchronize before release_after_sync()."""
        self._restore()

    def release_after_sync(self):
        require(not self.patches, "restore before release")
        self.views.clear(); self.layouts.clear(); self.stream_refs.clear(); self.released=True

    @property
    def failed(self):
        return self.failure is not None

    def safe_receipt(self):
        return self.report()

    def report(self):
        return {'counts':dict(self.counts),'first_failure':self.failure,
                'released':self.released,'actual_write_gpu_count':None,
                'uva_pool_lifetime_proven':False}


def comparator(torch):
    """Future runtime only. No pointer dereference/cast; integer data buffers only.
    Native tables are compared, never copied to host or serialized. torch.equal
    returns only booleans. Diagnostic allocations/synchronization affect timing.
    """
    def compare(ptrs,strides,tensors):
        expected_ptrs = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.uint64,device=ptrs.device)
        expected_strides = torch.tensor([t.stride(0) for t in tensors], dtype=torch.int64,device=strides.device)
        return bool(torch.equal(ptrs,expected_ptrs) and torch.equal(strides,expected_strides))
    return compare
