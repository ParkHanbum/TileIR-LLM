from gdn_integer_finite_typed_v1.core import PrefillCore as Typed
from lean26_gdn_finite_runtime_v1.core import FLOAT,INTEGER,ORDER
from gdn_integer_finite_typed_v1.qualifier import Qualifier
class PrefillCore(Typed):
 def start(self):
  super().start()
  import torch
  self.integer_qualifier=Qualifier(torch)
 def integer_check(self,site,t):
  # Integer native and candidate booleans are both True; shadow selects native.
  return self.integer_qualifier.check(site,t,mode="shadow" if self.controller.integer_phase=="shadow" else "replace")
 def finish(self):
  out=super().finish();out['gdn_integer_proof']=self.integer_qualifier.finish()
  out['gdn_integer_proof'].update(phase=self.controller.integer_phase,selected_native_predicates=72 if self.controller.integer_phase=='shadow' else 0,native_oracle_predicates=72)
  return out
