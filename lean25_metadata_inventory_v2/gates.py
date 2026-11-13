"""Existing exact endpoint gates plus metadata prerequisite; CPU only at import."""
import json
from pathlib import Path
from lean_public_runtime_qualification_v2.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch
from lean_public_runtime_qualification_v2.gates import qualification as numerical_qualification,audit as endpoint_audit
P=Path(__file__).resolve().parent.parent
def qualification(root):
 numerical_qualification(P/'lean_public_runtime_qualification_v2')
 q=json.loads((Path(root)/'HELPER-SPECS.json').read_text());assert q['complete'],'Helper metadata contracts pending'
 setup=json.loads((Path(root)/'SETUP-SPECS.json').read_text());assert setup['complete']
 for h in q['helpers']+setup['helpers']:assert sha(h['source_path'])==h['source_sha256']
 link=json.loads((P/'lean_public_runtime_qualification_v2/final_proof_01/PREREQUISITE-LINK.json').read_text())
 for k in ('result','launch','recovery'):assert sha(link[k]['path'])==link[k]['sha256']
 assert endpoint_audit(json.loads(Path(link['result']['path']).read_text()))['passed']
 return dict(lean_endpoint_proof=link,helper_specs=q)
def audit(r,root=None):
 root=Path(root or Path(__file__).resolve().parent);x=endpoint_audit(r,root)
 if not x['passed']:return x
 try:
  assert r['request_helpers_restored']
  setup=r['setup_metadata'];assert setup['helpers_restored'] and sha(setup['path'])==setup['sha256']
  m=r['metadata_inventory'];assert m['calls']>0 and m['sites']>0
  assert sha(m['metadata_path'])==m['metadata_sha256'] and sha(m['trace_path'])==m['trace_sha256']
  data=json.loads(Path(m['metadata_path']).read_text());assert data['helpers_restored'] and not data['failed']
  for h in json.loads((root/'HELPER-SPECS.json').read_text())['helpers']:
   n=data['helper_counts'][h['label']];assert n>=0
   if h.get('expected_calls') is not None:assert n==h['expected_calls']
  return dict(passed=True,scope='Exact endpoints plus bounded metadata diagnostic, not performance')
 except (AssertionError,KeyError,TypeError)as e:return dict(passed=False,errors=[repr(e)])
