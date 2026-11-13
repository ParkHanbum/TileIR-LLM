import inspect,json,hashlib,textwrap,functools,ast
from pathlib import Path
from .registry import Hold,require
from .diagnostic import Diagnostic,safe_code

def verify_bindings(mr,u):
 for s in json.loads(Path(__file__).with_name('SOURCE-BINDINGS.json').read_text()):
  module=mr if s['owner']=='GPUModelRunner' else u;fn=getattr(getattr(module,s['owner']),s['attribute'])
  require(hashlib.sha256(Path(s['source_path']).read_bytes()).hexdigest()==s['source_sha256'],'source_changed')
  require(fn.__module__==s['module'] and fn.__qualname__==s['owner']+'.'+s['attribute'],'binding_identity')
  require(list(inspect.signature(fn).parameters)==s['parameters'],'binding_signature')
  node=next(x for x in ast.parse(textwrap.dedent(inspect.getsource(fn))).body if isinstance(x,ast.FunctionDef))
  require(hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==s['binding_ast_sha256'],'binding_source')
 require(mr.KVBlockZeroer is u.KVBlockZeroer,'constructor_alias')

class Hooks:
 def __init__(self):self.diag=Diagnostic();self.phase='warmup';self.warmup_depth=0;self.bindings=[];self.warmup_attempted=0;self.warmup_returned=0;self.failed=None;self.restored=False;self.released=False
 def guard(self):require(self.failed is None and self.diag.failed is None,'latched_failure')
 def latch(self,e,stage):self.failed=safe_code(e,stage);return Hold(self.failed)
 def install(self,mr,u,validator=verify_bindings):
  self.guard()
  try:
   validator(mr,u)
   def bind(obj,name,new):
    old=getattr(obj,name);self.bindings.append((obj,name,old,new));setattr(obj,name,new);return old
   native_setup=mr.GPUModelRunner._init_kv_zero_meta;native_zero=u.KVBlockZeroer.zero_block_ids;native_warm=u.KVBlockZeroer.warmup
   def setup(runner):
    self.guard()
    try:return self.diag.setup(native_setup,runner,u.AttentionSpec,u.torch.Tensor)
    except BaseException as e:raise self.latch(e,'setup_failed')from None
   def warm(z,num_kv_blocks):
    self.guard()
    try:
     require(self.phase=='warmup' and self.warmup_depth==0,'warmup_phase');self.warmup_depth=1
     return native_warm(z,num_kv_blocks)
    except BaseException as e:raise self.latch(e,'warmup_failed')from None
    finally:self.warmup_depth=0
   def zero(z,ids):
    self.guard()
    try:
     if self.phase=='warmup':
      require(self.warmup_depth==1 and type(ids)is list and ids==[0],'unclassified_init_zero')
      self.warmup_attempted+=1;ret=native_zero(z,ids);self.warmup_returned+=1;require(ret is None,'native_return');return ret
     require(self.phase=='request','request_phase');return self.diag.zero(native_zero,z,ids)
    except BaseException as e:raise self.latch(e,'zero_boundary_failed')from None
   bind(mr.GPUModelRunner,'_init_kv_zero_meta',setup);bind(u.KVBlockZeroer,'warmup',warm);bind(u.KVBlockZeroer,'zero_block_ids',zero)
   return self
  except BaseException as e:
   self.failed=safe_code(e,'install_failed');self.close();raise Hold(self.failed)from None
 def begin_request(self):
  self.guard();require(self.phase=='warmup' and self.diag.setup_returned==1,'request_before_setup');self.phase='request'
 def receipt(self):
  return dict(phase=self.phase,failed=self.failed or self.diag.failed,setup_attempted=self.diag.setup_attempted,setup_returned=self.diag.setup_returned,warmup_zero_attempted=self.warmup_attempted,warmup_zero_returned=self.warmup_returned,request_zero_attempted=self.diag.native_zero_calls,request_zero_returned=self.diag.native_zero_returned,geometry=self.diag.receipt,candidate_attempted=self.diag.candidate_calls,candidate_returned=self.diag.candidate_returned,proof=self.diag.proof,restored=self.restored,released=self.released,owner_payload_bytes=self.diag.proof.get('planned_d2h_bytes',0),owner_payload_count_is_planned=True,table_metadata_bytes=192 if self.diag.receipt is not None else 0)
 def close(self):
  errors=[]
  for obj,name,old,new in reversed(self.bindings):
   if getattr(obj,name)is not new:errors.append('foreign_binding');continue
   setattr(obj,name,old)
  self.bindings=[];self.phase='cleanup';self.restored=not errors
  require(not errors,'restore_failed')
 def release(self,after_synchronize=False):
  require(self.restored,'release_before_restore');self.diag.close(after_synchronize);self.released=True
