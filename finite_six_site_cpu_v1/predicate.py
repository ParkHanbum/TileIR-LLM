"""Retain original caller exception and short-circuit expression semantics."""
PAYLOAD_ERROR='NaN/Inf outside finite-only contract'
def finite_boolean(x,require_finite):
 try:require_finite(x)
 except AssertionError as e:
  if str(e)==PAYLOAD_ERROR:return False
  raise
 return True
