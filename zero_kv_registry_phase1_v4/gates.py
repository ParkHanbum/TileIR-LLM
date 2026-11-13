import json,hashlib
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def qualification(root):
 spec=json.loads((Path(root)/'SOURCE-BINDINGS.json').read_text())
 for x in spec:assert sha(x['source_path'])==x['source_sha256']
 root=Path(root);links=json.loads((root/'PREREQUISITE.json').read_text())
 for p,h in links.items():assert sha(p)==h
 base=root.parent/'lean25_metadata_inventory_v2';result=json.loads((base/'leanMetadata02/RESULT.json').read_text());launch=json.loads((base/'leanMetadata02-launch.json').read_text());recovery=json.loads((base/'leanMetadata02-recovery.json').read_text())
 assert result['complete'] and type(launch['returncode'])is int and launch['returncode']==0 and launch['finished']
 assert type(recovery['actual_returncode'])is int and recovery['actual_returncode']==0 and recovery['child_absent'] and recovery['remaining_process_group']==[] and recovery['lock_available']
 assert recovery['result_sha256']==sha(base/'leanMetadata02/RESULT.json') and recovery['launch_sha256']==sha(base/'leanMetadata02-launch.json')
 return dict(public_source_pins=True,metadata_actual_exit0=True,phase1_only=True,actual_registry_pending=True)
def audit(r,root=None):
 try:
  d=r['diagnostic'];assert r['complete'] and r['checks_passed'] and not r['cleanup_errors'] and not r.get('hold')
  assert r['resolved_runtime']==dict(enforce_eager=True,compilation_mode=0,cudagraph_mode='NONE',async_scheduling=True,precision=dict(bf16=True,split_k=True))
  assert len(r['output_token_ids'])==32 and all(len(x)==16 for x in r['output_token_ids'])
  assert r['tokens_gate']['exact'] and r['public_gate']['exact_numeric'] and r['public_gate']['values']==2560
  assert d['setup_attempted']==d['setup_returned']==1 and d['request_zero_attempted']==d['request_zero_returned']==1
  assert d['geometry']['table_equal'] and d['geometry']['within_budget'] and d['owner_payload_bytes']==0 and d['table_metadata_bytes']==192
  assert d['restored'] and d['released'] and r['engine_shutdown']
  return dict(passed=True,phase1_only=True)
 except (AssertionError,KeyError,TypeError):return dict(passed=False,phase1_only=True)
