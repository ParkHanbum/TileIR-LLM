"""Combine reviewed standalone lean sources; combined endpoint is still pending."""
from residual_four_gdn_owned_lean_v2 import adapter as base
from lean26_gdn_finite_runtime_v1 import adapter as controller_module
ATTR=base.ATTR
ZERO=None

def install(model):
    result=base.install(model)
    if ZERO is None:raise RuntimeError('zero setup missing before engine')
    ZERO.controller=getattr(model,controller_module.ATTR)
    return result

def start(model,mode='lean'):
    result=base.start(model,mode=mode)
    try:ZERO.begin_request()
    except BaseException as e:ZERO.controller.latch('zero/start',e);raise
    return result

def finish(model):
    ZERO.guard()
    r=ZERO.receipt()
    if not (r['candidate_attempted']==r['candidate_returned']==1 and r['request_zero_attempted']==r['request_zero_returned']==0 and r['fresh_registry_after_native']):
        error=RuntimeError('zero request census');ZERO.controller.latch('zero/finish',error);raise error
    result=base.finish(model);result['zero_lean']=r;return result

def close(model):return base.close(model)
def release(model):return base.release(model)
