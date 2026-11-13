"""Bounded source-site raw shadow executor; NOT a production replacement.
No CUDA imports/queries at module scope. Coordinator supplies reviewed request.
"""
import hashlib,inspect,json
from contextlib import contextmanager
from pathlib import Path
from .contract import NAMES,READS,clone_inputs,validate_values

def _raw_equal(torch,a,b):
    return bool(torch.equal(a.view(-1).view(torch.uint8),b.view(-1).view(torch.uint8)))

def _launch_owned(torch,owned):
    import cuda.tile as ct
    from .kernels import post_update_kernel
    flat=tuple(x.view(-1) for x in owned)
    with torch.cuda.device(owned[0].device):
        ct.launch(torch.cuda.current_stream(owned[0].device),(32,1,1),post_update_kernel,flat+(1,128,248320,True,True))

class PostShadow:
    def __init__(self,original,torch,limit=1):
        if type(limit)is not int or not 1<=limit<=16:raise ValueError('shadow limit1..16')
        self.original=original;self.torch=torch;self.limit=limit
        self.signature=inspect.signature(original)
        if tuple(self.signature.parameters)!=NAMES:raise ValueError('Source signature changed')
        self.calls=0;self.native_attempts=0;self.native_returns=0;self.candidate_attempts=0;self.candidate_enqueues=0;self.shadow_passed=0;self.failed=False
    def __call__(self,*args,**kwargs):
        if self.failed:raise RuntimeError('Failed diagnostic cannot continue')
        self.calls+=1
        if self.shadow_passed>=self.limit:
            self.native_attempts+=1
            result=self.original(*args,**kwargs);self.native_returns+=1;return result
        bound=self.signature.bind(*args,**kwargs); live=tuple(bound.arguments[n] for n in NAMES)
        try:
            owned=clone_inputs(self.torch,live)
            validate_values(self.torch,owned)
            immutable=tuple((i,owned[i].clone()) for i in READS)
            # No candidate touches live state. Original is called exactly once.
            self.native_attempts+=1
            result=self.original(*args,**kwargs);self.native_returns+=1
            self.candidate_attempts+=1
            _launch_owned(self.torch,owned)
            self.candidate_enqueues+=1
            self.torch.cuda.current_stream(live[0].device).synchronize()
            if result is not None:raise AssertionError('Original return contract changed')
            if not all(_raw_equal(self.torch,a,b) for a,b in zip(live,owned)):raise AssertionError('Full tensor raw mismatch')
            if not all(_raw_equal(self.torch,live[i],before) and _raw_equal(self.torch,owned[i],before) for i,before in immutable):raise AssertionError('Read-only input mutation')
            self.shadow_passed+=1
            return result
        except BaseException:
            self.failed=True
            raise # No fallback/retry after uncertain native or candidate submission.
    def snapshot(self):
        return dict(calls=self.calls,native_attempts=self.native_attempts,candidate_attempts=self.candidate_attempts,candidate_enqueues=self.candidate_enqueues,native_returns=self.native_returns,shadow_passed=self.shadow_passed,limit=self.limit,failed=self.failed,scope='dense device clone arithmetic only; not live UVA qualification; no timing')

@contextmanager
def post_update_shadow(limit=1):
    import torch
    from vllm.v1.worker.gpu import model_runner
    pins=json.loads((Path(__file__).parent/'SOURCE-PINS.json').read_text())
    for p,h in pins.items():
        if p.endswith('.py') and hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise RuntimeError('Public source changed')
    original=model_runner.post_update
    # Refuse wrapping someone else's observer/adapter or broad/global patch.
    source=inspect.getsourcefile(original)
    if source is None or not source.endswith('/v1/worker/gpu/input_batch.py'):raise RuntimeError('Unexpected callable owner')
    wrapper=PostShadow(original,torch,limit);model_runner.post_update=wrapper
    try:yield wrapper
    finally:
        current=model_runner.post_update
        model_runner.post_update=original
        if current is not wrapper:raise RuntimeError('Concurrent helper binding changed; original restored')
