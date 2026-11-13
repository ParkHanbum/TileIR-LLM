"""Owned public Python gate variants. Never relabel a typed receipt as native."""
import ast,inspect,textwrap

def copy_function(fn,globals_update=None,field_changes=None):
 tree=ast.parse(textwrap.dedent(inspect.getsource(fn)))
 assert len(tree.body)==1 and isinstance(tree.body[0],ast.FunctionDef)
 changes=field_changes or {};counts=dict.fromkeys(changes,0)
 class Fields(ast.NodeTransformer):
  def visit_Constant(self,n):
   if isinstance(n.value,str) and n.value in changes:
    counts[n.value]+=1;return ast.copy_location(ast.Constant(changes[n.value]),n)
   return n
 tree=ast.fix_missing_locations(Fields().visit(tree))
 assert all(n==1 for n in counts.values()),'Exact original field references required'
 ns=dict(fn.__globals__);ns.update(globals_update or {})
 exec(compile(tree,'<owned-integer72-gate>','exec'),ns)
 return ns[fn.__name__]

def typed_contract(d):
 p=d['components']['gdn_prefill_core']['gdn_finite']
 assert 'retained_integer_checks'not in p and 'retained_integer_site_counts'not in p
 assert p['typed_integer_checks']==72 and p['typed_integer_site_counts']==dict.fromkeys(('cu_seqlens','chunk_indices','chunk_offsets'),24)
 assert p['integer_gpu_kernels']==0 and p['integer_stream_synchronizations']==72 and p['integer_native_predicates']==0
 assert p['float_checks']==360

_CACHE=None
def build():
 global _CACHE
 if _CACHE is not None:return _CACHE
 from lean26_gdn_finite_runtime_v1 import gates as base
 from residual_four_gdn_owned_lean_v2 import gates as gdn
 from residual_four_gdn_zero_lean_v3 import gates as zero
 changed=copy_function(base.dispatch,field_changes={'retained_integer_site_counts':'typed_integer_site_counts','retained_integer_checks':'typed_integer_checks'})
 def dispatch(d):typed_contract(d);return changed(d)
 endpoint=copy_function(base.audit,{'dispatch':dispatch})
 gdn_audit=copy_function(gdn.audit,{'base_audit':endpoint})
 zero_audit=copy_function(zero.audit,{'lean_audit':gdn_audit})
 _CACHE=(dispatch,zero_audit);return _CACHE

def dispatch(d):return build()[0](d)
def audit(r,root):return build()[1](r,root)
