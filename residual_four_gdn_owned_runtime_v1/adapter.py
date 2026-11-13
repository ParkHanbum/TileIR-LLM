"""Existing base26/472 + four64 + GDN1152, separate census; no numerical edits."""
from lean26_gdn_finite_runtime_v1 import adapter as base
from tileir_gdn_owned_replacement_v6 import runtime_adapter as owned
from residual_four_lean26_runtime_v1 import adapter as four_module
from .guard import connect
ATTR=base.ATTR

def install(model):
    r=base.install(model)
    try:r['owned_replacement']=owned.install(model)
    except BaseException as e:
        getattr(model,ATTR).latch('gdn_owned/install',e);raise
    return r

def start(model,mode='lean'):
    controller=getattr(model,ATTR)
    try:
        r=base.start(model,mode=mode)
        owned.start(model,mode='replace')
        four=four_module.CURRENT
        if four is None:raise RuntimeError('four missing')
        connect(getattr(model,owned.ATTR).ops,controller,four)
        return r
    except BaseException as e:controller.latch('gdn_owned/start',e);raise

def finish(model):
    controller=getattr(model,ATTR)
    try:
        if controller.failed is not None:raise RuntimeError('combined latch')
        g=owned.finish(model)
        r=base.finish(model);r['owned_replacement']=g
        return r
    except BaseException as e:controller.latch('gdn_owned/finish',e);raise

def close(model):
    errors=[];g=None
    if hasattr(model,owned.ATTR):
        try:g=owned.close(model)
        except BaseException as e:errors.append(type(e).__name__)
    try:r=base.close(model)
    except BaseException as e:errors.append(type(e).__name__);r={}
    if errors:raise RuntimeError('combined close HOLD '+','.join(errors))
    r['owned_replacement']=g;return r

def release(model):
    # Existing four cleanup retains UVA owners until outer final sync/engine shutdown.
    if hasattr(model,owned.ATTR):
        c=getattr(model,owned.ATTR)
        if c.ops is not None:c.ops.release_after_sync()
        delattr(model,owned.ATTR)
    return base.release(model)
