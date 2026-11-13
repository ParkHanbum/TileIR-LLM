from lean26_gdn_finite_runtime_v1 import adapter as lean
from . import integer
ATTR=lean.ATTR
ORDER=lean.ORDER
SPECS=lean.SPECS
def install_integer(model,output):return integer.install(model,output)
def install(model):return lean.install(model)
def start_shadow(model):return integer.start(model,'shadow')
def finish_shadow(model):return integer.finish(model)
def start(model,mode='lean'):
 integer.start(model,'replace');return lean.start(model,mode)
def finish(model):
 result=lean.finish(model);result['sampler_integer']=integer.finish(model);return result
CLOSE_STATE='_sampler_integer_close_v2'
def close(model):
 # Keep both owner states available for diagnosis/retry if either restoration fails.
 result=None;integer_result=None;errors=[]
 try:
  result=lean.close(model);assert isinstance(result,dict)and result.get('restored')is True
 except BaseException as e:errors.append(('lean',repr(e)))
 try:
  integer_result=integer.close(model);assert isinstance(integer_result,dict)and integer_result.get('restored')is True
 except BaseException as e:errors.append(('sampler_integer',repr(e)))
 state=dict(lean=result,sampler_integer=integer_result,errors=errors,restored=not errors)
 setattr(model,CLOSE_STATE,state)
 if errors:raise RuntimeError('Independent adapter restore failures',errors)
 assert isinstance(result,dict)and integer_result['restored']
 result['sampler_integer_restore']=integer_result;return result
def release(model):
 state=getattr(model,CLOSE_STATE,None)
 assert state is not None and state['restored']and not state['errors'],'Both independent restores required before owner release'
 lean.release(model);integer.release(model);delattr(model,CLOSE_STATE)
