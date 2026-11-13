"""Profile-only public API ranges. Absent from all wall samples; no numerical changes."""
from contextlib import contextmanager
import functools
@contextmanager
def public_api_scopes(torch,enabled):
 if not enabled:yield;return
 from lean_public_composition_v1.support import Base
 from lean_public_composition_v2.adapter import ORDER
 old=Base.call
 @functools.wraps(old)
 def scoped(self,*args,**kwargs):
  assert self.key in ORDER
  with torch.profiler.record_function('tileir_public_api::'+self.key):return old(self,*args,**kwargs)
 Base.call=scoped
 try:yield
 finally:
  assert Base.call is scoped,'Foreign API scope binding; do not silently overwrite'
  Base.call=old
