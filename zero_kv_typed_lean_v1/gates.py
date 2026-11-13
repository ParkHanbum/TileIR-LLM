import json,hashlib
from pathlib import Path
def qualification(root):
 root=Path(root)
 for p,h in json.loads((root/'PREREQUISITE.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 q=root.parent/'zero_kv_typed_runtime_v2';r=json.loads((q/'zeroKVTypedRuntime01/RESULT.json').read_text());l=json.loads((q/'zeroKVTypedRuntime01-launch.json').read_text());v=json.loads((q/'zeroKVTypedRuntime01-recovery.json').read_text())
 from zero_kv_typed_runtime_v2.gates import audit
 assert audit(r)['passed'] and type(l['returncode'])is int and l['returncode']==0 and l['finished']
 assert v['actual_returncode']==0 and v['child_absent'] and not v['remaining_process_group'] and v['lock_available']
 return dict(actual_payload_proof_pass=True,actual_model_pending=True)
def audit(r,root=None):
 try:
  d=r['diagnostic'];p=d['proof'];g=d['geometry']
  assert r['complete'] and r['checks_passed'] and not r['cleanup_errors'] and not r.get('hold')
  assert r['tokens_gate']['exact'] and r['public_gate']['exact_numeric'] and r['public_gate']['values']==2560
  assert d['setup_attempted']==d['setup_returned']==1 and d['request_zero_attempted']==d['request_zero_returned']==0
  assert d['candidate_attempted']==d['candidate_returned']==1 and g['table_equal'] and g['readonly_tables_ids_exact']
  assert p['owner_payload_read_bytes']==0 and p['owner_hashes_computed']==0 and p['target_oracle_executed']is False and p['outside_oracle_executed']is False
  assert p['zero_attempted']==p['zero_returned']==p['expected_source_launches']>0
  assert g['segments']==8 and g['allocations']==1 and g['unique_block_count']==32 and g['block_count']==32 and g['selected_union_bytes']==553648128
  assert p['target_bytes']==g['selected_union_bytes'] and 2*p['allocation_bytes']-p['target_bytes']<=16*1024**3
  assert d['restored'] and d['released'] and r['engine_shutdown'] and r['synchronized']
  return dict(passed=True,actual_model=True,performance=False)
 except (AssertionError,KeyError,TypeError):return dict(passed=False,actual_model=True)
