"""Compose unchanged current lean26 lifecycle with independent owned-copy shadow."""
from lean26_gdn_finite_runtime_v1 import adapter as base
from . import runtime_adapter as owned
ATTR=base.ATTR

def install(model):
    original=base.install(model)
    diagnostic=owned.install(model)
    original['owned_replacement']=diagnostic
    return original

def start(model,mode='lean'):
    original=base.start(model,mode=mode)
    owned.start(model,mode='replace')
    return original

def finish(model):
    diagnostic=owned.finish(model)
    result=base.finish(model)
    result['owned_replacement']=diagnostic
    return result

def close(model):
    errors=[]; results={}
    if hasattr(model,owned.ATTR):
        try: results['owned_replacement']=owned.close(model)
        except BaseException as e: errors.append(type(e).__name__)
    try: results['base']=base.close(model)
    except BaseException as e: errors.append(type(e).__name__)
    if errors: raise RuntimeError('combined close failed: '+','.join(errors))
    result=results['base']; result['owned_replacement']=results.get('owned_replacement')
    return result

def release(model):
    # Existing final_cleanup calls this only after close AND both synchronizations.
    if hasattr(model,owned.ATTR):
        c=getattr(model,owned.ATTR)
        if c.ops is not None: c.ops.release_after_sync()
        delattr(model,owned.ATTR)
    return base.release(model)
