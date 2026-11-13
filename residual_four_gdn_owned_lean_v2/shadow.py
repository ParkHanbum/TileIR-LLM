"""CPU-prepared same-input shadow. No framework import at module scope."""
import ast
import copy
import types
from tileir_gdn_owned_copies_v1.transform import transform_forward

class ShadowOps:
    """Lean operational metadata checks, NOT per-call raw/numerical proof."""
    def __init__(self,torch,candidate):
        self.t,self.c=torch,candidate
        self.calls=self.native_returns=self.candidate_returns=self.passed=0
        self.failed=False;self.expected='zero';self.outputs=[]
    def _begin(self,key):
        if self.failed or key!=self.expected:
            self.failed=True;raise RuntimeError('lean owned scope/order')
        self.calls+=1
        if key=='zero':self.outputs.clear()
    def _fresh(self,a,b):
        s,t=a.untyped_storage(),b.untyped_storage()
        if not (s.data_ptr()+s.nbytes()<=t.data_ptr() or t.data_ptr()+t.nbytes()<=s.data_ptr()):
            raise RuntimeError('lean output overlap')
    def _output(self,out):
        for old in self.outputs:self._fresh(old,out)
        self.outputs.append(out) # Only current zero/b/a trio, not full-model proof retention.
    def zeros(self,shape,*,dtype,device):
        self._begin('zero')
        try:
            out=self.c.zeros(shape,dtype=dtype,device=device);self.candidate_returns+=1
            self._output(out);self.passed+=1;self.expected='b';return out
        except BaseException:self.failed=True;raise
    def _copy(self,x,key):
        self._begin(key)
        try:
            from tileir_gdn_owned_copies_v1.api import check_copy
            check_copy(x,0 if key=='b' else 32,self.t)
            if x.untyped_storage().nbytes()!=x.shape[0]*64*2:raise RuntimeError('full backing extent')
            signature=(id(x),x.data_ptr(),x.untyped_storage().data_ptr(),x.storage_offset(),tuple(x.shape),tuple(x.stride()))
            out=getattr(self.c,'contiguous_'+key)(x);self.candidate_returns+=1
            assert signature==(id(x),x.data_ptr(),x.untyped_storage().data_ptr(),x.storage_offset(),tuple(x.shape),tuple(x.stride()))
            self._fresh(x,out);self._output(out);self.passed+=1
            self.expected='a' if key=='b' else 'zero';return out
        except BaseException:self.failed=True;raise
    def contiguous_b(self,x):return self._copy(x,'b')
    def contiguous_a(self,x):return self._copy(x,'a')
    def receipt(self):
        return dict(calls=self.calls,native_target_calls=0,candidate_returns=self.candidate_returns,metadata_passes=self.passed,failed=self.failed,order_complete=self.expected=='zero',returns_candidate=True,raw_rechecked=False,cpu_oracle_calls=0)
    def release_after_sync(self):self.outputs.clear()


def owned_function(source, native_function, ops):
    """Called only after pinned source check and exact cached-binding audit."""
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QwenGatedDeltaNetAttention')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward_cuda')
    if node.decorator_list: raise RuntimeError('unexpected forward decorators')
    node=transform_forward(node)
    # Avoid evaluating annotations from copied source. Runtime behavior is unchanged.
    node.returns=None
    for arg in (*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs): arg.annotation=None
    namespace=dict(native_function.__globals__); namespace['_owned_gdn_copies']=ops
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),native_function.__code__.co_filename,'exec'),namespace)
    return namespace['forward_cuda']

class Binding:
    """Install on supplied actual Qwen attention owners, not ChunkGatedDeltaRule.
    Caller obtains owners from model.named_modules and provides public exact class.
    Existing class/instance forward edits are rejected rather than overwritten.
    """
    def __init__(self, owners, authority, owned):
        self.owners=tuple(owners); self.authority=authority; self.owned=owned; self.saved=[]; self.invocations=[0]*len(self.owners)
    def install(self):
        if len(self.owners)!=24 or len({id(x) for x in self.owners})!=24: raise RuntimeError('owner census')
        # Complete preflight before modifying any instance.
        for o in self.owners:
            f=o._forward_method
            if not (getattr(f,'__self__',None) is o and getattr(f,'__func__',None) is self.authority and o.forward_cuda.__func__ is self.authority):
                raise RuntimeError('cached forward binding not exact authority')
            if o.gqa_interleaved_layout or o.enable_fused_gdn_spec_decode:
                raise RuntimeError('unsupported forward branch')
        for i,o in enumerate(self.owners):
            def counted(owner,*args,_index=i,**kwargs):
                self.invocations[_index]+=1
                return self.owned(owner,*args,**kwargs)
            old=o._forward_method; new=types.MethodType(counted,o)
            self.saved.append((o,old,new)); o._forward_method=new
    def restore(self):
        conflict=False
        for o,old,new in reversed(self.saved):
            if o._forward_method is not new: conflict=True; continue
            o._forward_method=old
        if conflict: raise RuntimeError('binding restore conflict')
        self.saved.clear()


def qualification(model, attention_class, source, torch, ct, kernels):
    """Return context manager for existing reviewed single-request engine harness.
    Caller verifies pinned source bytes before call; no engine construction here.
    """
    from contextlib import contextmanager
    from tileir_gdn_owned_copies_v1.api import OwnedCopies
    @contextmanager
    def context():
        owners=[m for _,m in model.named_modules() if type(m) is attention_class]
        ops=ShadowOps(torch,OwnedCopies(torch,ct,kernels))
        owned=owned_function(source,attention_class.forward_cuda,ops)
        binding=Binding(owners,attention_class.forward_cuda,owned)
        binding.install()
        try:
            yield ops
            if binding.invocations != [16]*24 or ops.passed!=1152 or ops.failed or ops.expected!='zero':
                raise RuntimeError('incomplete actual shadow census')
        except BaseException:
            ops.failed=True
            raise
        finally:
            binding.restore()
            # If synchronization fails, retained tensors stay referenced by ops.
            torch.cuda.synchronize()
            ops.release_after_sync()
    return context()
