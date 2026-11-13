"""Delegate original combined adapter; pair install precedes base and uses its Bridge only."""
from copy import deepcopy
from residual_four_gdn_zero_lean_v3 import adapter as base
ATTR=base.ATTR
EXT=None
OUTPUT=None
ZERO=None

def install(model):
 base.ZERO=ZERO
 assert model is EXT.model and EXT.pair is not None and EXT.reference_observed is not None
 from .predicate_controller import Controller, ATTR as controller_attr
 from residual_four_gdn_owned_lean_v2 import runtime_adapter as owned
 assert not hasattr(model,controller_attr)
 controller=Controller(model);setattr(model,controller_attr,controller)
 try:
  result=controller.install();result['owned_replacement']=owned.install(model)
  if ZERO is None:raise RuntimeError('zero setup missing before engine')
  ZERO.controller=controller;return result
 except BaseException as e:controller.latch('install',e);raise
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
 r=EXT.pair.close(lambda:base.close(model))
 # Do not insert the parent receipt into its own components.lean dictionary.
 result=deepcopy(r['components']['lean']);result['integer_pair_restore']=deepcopy(r);return result

def release(model):
 from integer_pair_runtime_v1 import adapter as pair
 if EXT.pair is not None:EXT.pair.require_release()
 base.release(model)
 pair.integer.release(model);pair.cast.release(model)
 if hasattr(model,pair.PAIR):delattr(model,pair.PAIR)
