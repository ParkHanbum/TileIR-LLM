"""Local copied apply, finite predicates only; actual descriptor/census observation."""
import inspect,textwrap,json,hashlib
from pathlib import Path
from lean_public_composition_v2.core_parent import PrefillCore as Original
R=Path(__file__).parent
ORDER=('q','k','v','g','beta','initial_state','cu_seqlens','chunk_indices','chunk_offsets','g_cumsum','A_kkt','A_inverse','w','u','h','v_new','final_state','o')
INTEGER=('cu_seqlens','chunk_indices','chunk_offsets')
FLOAT=tuple(k for k in ORDER if k not in INTEGER)
class PrefillCore(Original):
 def install(self,model):
  spec=json.loads((R/'SOURCE.json').read_text());assert hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()==spec['sha256']
  old=textwrap.dedent(inspect.getsource(Original.apply)).strip()
  candidate=(R/'apply.candidate.py.txt').read_text().strip()
  assert candidate==old.replace('bool(torch.isfinite(t).all())','_gdn_finite(t)')
  ns=dict(Original.apply.__globals__);ns['_gdn_finite']=self.finite
  exec(compile(candidate,str(R/'apply.candidate.py.txt'),'exec'),ns);self.body=ns['apply']
  self.finite_scope=None;self.site_counts={k:0 for k in ORDER};self.descriptors={}
  return super().install(model)
 def start(self):
  super().start();assert self.finite_scope is None;self.site_counts={k:0 for k in ORDER};self.descriptors={}
 def finite(self,t):
  import torch
  from tileir_validation_primitives_v3.api import all_finite
  assert self.controller.active and self.controller.failed is None and self.finite_scope is not None
  i=self.finite_scope;assert 0<=i<len(ORDER);site=ORDER[i];assert self.site_counts[site]<24
  # Original apply checks input descriptors before this point. Outputs are
  # post-producer checked; the unchanged descriptor checks follow finite.
  meta=dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),offset=t.storage_offset(),numel=t.numel())
  if site in self.descriptors:assert meta==self.descriptors[site],('GDN finite descriptor changed',site)
  else:self.descriptors[site]=meta
  if site in INTEGER:
   assert t.dtype in (torch.int32,torch.int64);ok=bool(torch.isfinite(t).all())
  else:
   assert t.dtype in (torch.bfloat16,torch.float32);ok=all_finite(t)
  self.site_counts[site]+=1;self.finite_scope=i+1;return ok
 def apply(self,v):
  try:
   self.guard();assert self.finite_scope is None;self.finite_scope=0
   out=self.body(self,v);assert self.finite_scope==18;return out
  except BaseException as e:self.controller.latch(self.key,e);raise
  finally:self.finite_scope=None
 def finish(self):
  out=super().finish();assert self.finite_scope is None and self.site_counts==dict.fromkeys(ORDER,24)
  out['gdn_finite']=dict(float_site_counts={k:self.site_counts[k]for k in FLOAT},retained_integer_site_counts={k:self.site_counts[k]for k in INTEGER},float_checks=360,retained_integer_checks=72,actual_descriptors=self.descriptors,host_scalar_status_retained=True,scope='Observed candidate source boundaries; not physical kernel counts')
  return out
