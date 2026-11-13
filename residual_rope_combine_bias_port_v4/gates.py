"""CPU-only evidence/result gates. All three candidates remain experimental."""
import json
from pathlib import Path
from lean_public_runtime_qualification_v2.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch
from lean_public_runtime_qualification_v2.gates import qualification as prior_qualification,audit as endpoint_audit
P=Path(__file__).resolve().parent.parent
def qualification(root):
 prior=prior_qualification(P/'lean_public_runtime_qualification_v2')
 pins=json.loads((Path(root)/'METADATA-PINS.json').read_text())
 for p,h in pins.items():assert sha(p)==h,p
 doc=json.loads((P/'lean25_metadata_inventory_v2/leanMetadata02/METADATA.json').read_text())
 assert not doc['failed'] and doc['helpers_restored']
 for label in ('prepare_rope_positions','combine_sampled_and_draft_tokens','bias'):assert doc['helper_counts'][label]==16
 return dict(prior=prior,metadata=pins,scope='Descriptor evidence only; live values must pass guards')
def audit(r,root=None):
 root=Path(root or Path(__file__).resolve().parent);e=endpoint_audit(r,root)
 if not e['passed']:return e
 try:
  s=r['residual_shadow'];assert s['complete'] and s['restored'] and not s['failed'] and not r.get('shadow_cleanup_error')
  assert s['calls']==dict(prepare_rope_positions=16,combine_sampled_and_draft_tokens=16,bias=16)
  assert len(s['rows'])==48 and all(x['passed'] and x['native_calls']==x['candidate_calls']==1 and x['original_return_preserved'] for x in s['rows'])
  post=r['post_update_shadow'];assert r['post_update_restored'] and not post['failed'] and post['limit']==16
  assert all(post[k]==16 for k in ('calls','native_attempts','native_returns','candidate_attempts','candidate_enqueues','shadow_passed'))
  return dict(passed=True,scope='48 helper shadows plus16 post-update dense-clone shadows; native returns retained, no replacement/performance/ABI claim')
 except (AssertionError,KeyError,TypeError)as e:return dict(passed=False,errors=[repr(e)])
