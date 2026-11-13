"""Lifecycle plug-in for existing reviewed engine runner. No engine or GPU on import."""
from pathlib import Path
import ast
import hashlib
import inspect
import json
import textwrap
from .shadow import qualification
ATTR='_gdn_owned_lean_v2'
PATTERNS=(0x0000,0x8000,0x7f80,0xff80,0x7fc1,0x7f81,0xffc5,0x0001,0x007f,0x8001,0x3f80)

def source_checked(authority):
    root=Path(__file__).resolve().parent.parent/'tileir_gdn_owned_copies_v1'
    pins=json.loads((root/'SOURCE-PINS.json').read_text())
    for name,digest in pins.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest: raise RuntimeError('source pin mismatch')
    path=next(Path(n) for n in pins if n.endswith('/qwen_gdn_linear_attn.py'))
    source=path.read_text(); tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QwenGatedDeltaNetAttention')
    expected=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward_cuda')
    # Same exact source-slice/dedent path on both sides, including docstring whitespace.
    expected_text=''.join(source.splitlines(keepends=True)[expected.lineno-1:expected.end_lineno])
    expected_ast=ast.parse(textwrap.dedent(expected_text)).body[0]
    fn=getattr(authority,'__func__',authority)
    actual_text=inspect.getsource(fn)
    actual=ast.parse(textwrap.dedent(actual_text)).body[0]
    if (fn.__module__!='vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn'
        or fn.__qualname__!='QwenGatedDeltaNetAttention.forward_cuda'
        or str(inspect.signature(fn))!='(self, hidden_states: torch.Tensor) -> torch.Tensor'
        or fn.__code__.co_firstlineno!=expected.lineno
        or fn.__closure__ or ast.dump(expected_ast)!=ast.dump(actual)
        or textwrap.dedent(expected_text)!=textwrap.dedent(actual_text)
        or Path(inspect.getfile(fn))!=path):
        raise RuntimeError('runtime forward source mismatch')
    return source


class Controller:
    def __init__(self,model):
        self.model=model; self.failed=False; self.ctx=None; self.ops=None; self.closed=False
        self.originals=[]; self.child_originals=[]; self.synthetic=None; self.ready=False
    def install(self):
        import torch
        import cuda.tile as ct
        from tileir_gdn_owned_copies_v1 import kernels
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import QwenGatedDeltaNetAttention
        authority=QwenGatedDeltaNetAttention.forward_cuda
        self.source=source_checked(authority)
        owners=[m for _,m in self.model.named_modules() if type(m) is QwenGatedDeltaNetAttention]
        if len(owners)!=24: raise RuntimeError('owner census')
        self.originals=[(m,m._forward_method) for m in owners]
        # Preserve every other existing cached binding (including current GDN core adapter).
        self.child_originals=[(m,m._forward_method) for _,m in self.model.named_modules() if hasattr(m,'_forward_method') and all(m is not o for o in owners)]
        from .prerequisite import verify_shadow
        self.synthetic=verify_shadow()
        self.dependencies=(torch,ct,kernels,QwenGatedDeltaNetAttention)
        self.ready=True
        return dict(owners=24,synthetic=self.synthetic,mode='replacement qualification only')
    def start(self,mode):
        if self.failed or not self.ready or self.ctx is not None or mode!='replace': raise RuntimeError('invalid shadow state')
        torch,ct,kernels,cls=self.dependencies
        self.ctx=qualification(self.model,cls,self.source,torch,ct,kernels)
        try: self.ops=self.ctx.__enter__()
        except BaseException: self.failed=True; raise
        return dict(active=True)
    def finish(self):
        if self.failed or self.ctx is None or self.ops is None: raise RuntimeError('shadow inactive')
        try:
            self.ctx.__exit__(None,None,None); self.ctx=None
            assert all(m._forward_method is f for m,f in self.originals+self.child_originals),'binding restore identity'
            return dict(replacement_pass=True,operations=self.ops.receipt(),original_bindings_restored=True,core_bindings_unchanged=True)
        except BaseException: self.failed=True; raise
    def close(self):
        if self.ctx is not None:
            ctx=self.ctx; self.ctx=None
            # Force unsuccessful-context exit: do not falsely certify incomplete request.
            error=RuntimeError('engine request aborted')
            try: ctx.__exit__(RuntimeError,error,None)
            except RuntimeError as e:
                if e is not error: raise
        assert all(m._forward_method is f for m,f in self.originals+self.child_originals),'after-close identity'
        self.closed=True
        return dict(closed=True,failed=self.failed or bool(self.ops and self.ops.failed),operations=None if self.ops is None else self.ops.receipt())

def install(model,output=None):
    c=Controller(model); setattr(model,ATTR,c)
    try:return c.install()
    except BaseException:c.failed=True;raise

def start(model,mode='replace'): return getattr(model,ATTR).start(mode)
def finish(model): return getattr(model,ATTR).finish()
def close(model): return getattr(model,ATTR).close()
