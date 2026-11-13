from lean26_gdn_finite_runtime_v1 import adapter as lean
from . import cast
ATTR=lean.ATTR
ORDER=lean.ORDER
SPECS=lean.SPECS
def install_cast(model,output):return cast.install(model,output)
def install(model):return lean.install(model)
def start_shadow(model):return cast.start(model,'shadow')
def finish_shadow(model):return cast.finish(model)
def start(model,mode='lean'):
 cast.start(model,'replace');return lean.start(model,mode)
def finish(model):
 result=lean.finish(model);result['embedding_cast']=cast.finish(model);return result
def close(model):
 result=lean.close(model);result['embedding_cast_restore']=cast.close(model);return result
def release(model):lean.release(model);cast.release(model)
