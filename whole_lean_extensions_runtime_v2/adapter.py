"""Delegate original combined adapter; pair install precedes base and uses its Bridge only."""
from residual_four_gdn_zero_lean_v3 import adapter as base
ATTR=base.ATTR
EXT=None
OUTPUT=None
ZERO=None

def install(model):
 base.ZERO=ZERO
 assert model is EXT.model and EXT.pair is not None and EXT.reference_observed is not None
 return base.install(model)
def install_reference(model):return EXT.pair_before_base(model,OUTPUT)
def start_reference(model):
 assert model is EXT.model
 return EXT.reference_start()
def finish_reference(model):
 assert model is EXT.model
 return EXT.reference_finish()
def start(model,mode='lean'):
 try:
  EXT.start_new();return base.start(model,mode)
 except BaseException as e:EXT.bus.latch('start',e);raise

def finish(model):
 EXT.bus.guard()
 # New-domain finish before base flips active=False; no numeric operation here.
 new=EXT.finish_new();result=base.finish(model);result['new_domains']=new;return result

def close(model):
 if EXT.pair is None:return base.close(model)
 r=EXT.pair.close(lambda:base.close(model));result=r['components']['lean'];result['integer_pair_restore']=r;return result

def release(model):
 from integer_pair_runtime_v1 import adapter as pair
 if EXT.pair is not None:EXT.pair.require_release()
 base.release(model)
 pair.integer.release(model);pair.cast.release(model)
 if hasattr(model,pair.PAIR):delattr(model,pair.PAIR)
