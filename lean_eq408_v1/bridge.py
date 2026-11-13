"""Explicit source-owned bridges; coordinator owns install/restore order."""
from public_eq_all_guard_v2.owned import prepare
from lean_finite_504_v1.bridge import attention_body

def combined_attention(original,finite_checks,eq_checks):
 return attention_body(original,finite_checks.finite,eq_checks.equal_one)

class GDNBodyBinding:
 def __init__(self,part,eq_checks):
  self.part=part;self.original=part.body;self.installed=False
  # Verifies actual active finite body path/SHA/firstline/dedent AST and keeps _gdn_finite globals.
  self.replacement=prepare(self.original,'gdn_active',eq_checks.initial_zero)
 def install(self):
  assert not self.installed and self.part.body is self.original
  self.part.body=self.replacement;self.installed=True
 def close(self):
  if self.installed:
   assert self.part.body is self.replacement,'GDN body identity conflict'
   self.part.body=self.original;self.installed=False
  return dict(restored=True)
