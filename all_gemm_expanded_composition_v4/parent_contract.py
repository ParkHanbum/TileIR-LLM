"""Parent memory accounting does not relabel a wrapped core as native."""
ORDER=('gather','invert','mask','core_enter','core_exit','scatter')
def parent_gate(d):
 try:
  assert d['metadata_only'] is False and d['owner_payload_bytes']==0 and d['failed'] is None
  assert d['mode'] in ('shadow','replace') and d['core_native'] is False and d['core_delegated'] is True
  assert d['native_only'] is False and d['parent_targets_native'] is (d['mode']=='shadow') and d['target_oracles']==0
  assert d['native_invert_retained'] is (d['mode']=='shadow')
  assert d['supported_calls']==24 and d['logical_suboperation_calls']==96 and d['parent_api_calls']==96
  assert d['supported_vendor_fallback_calls']==0 and d['core_native_calls']==(24 if d['mode']=='shadow' else 0) and d['core_tileir_calls']==24
  rows=d['rows'];assert len(rows)==24 and len({x['layer'] for x in rows})==24
  assert all(x['passed'] is True and x['order']==list(ORDER) for x in rows)
  assert all(x['native_target_calls']==(3 if d['mode']=='shadow' else 0) and x['tile_target_calls']==(0 if d['mode']=='shadow' else 3) and x['native_invert_calls']==(1 if d['mode']=='shadow' else 0) and x['tile_invert_calls']==(0 if d['mode']=='shadow' else 1) for x in rows)
  links=d['core_links'];assert len(links)==24 and {x['parent'] for x in links}=={x['layer'] for x in rows}
  assert all(x['core_path']==x['parent']+'.chunk_gated_delta_rule' and x['exact_installed_guard'] and x['captured_by_parent'] for x in links)
  cr=d['core_phase_rows'];assert len(cr)==24 and {x['path'] for x in cr}=={x['core_path'] for x in links}
  assert all(x['passed'] and x['phase']=='prefill' for x in cr)
  return True
 except (AssertionError,KeyError,TypeError):return False

def verify_qualified_parent(root):
 from pathlib import Path
 import json,hashlib
 e=json.loads((Path(root)/'PARENT-PREREQUISITE.json').read_text())
 assert e['complete'] is True,'Parent v2 actual PASS pending; cannot install or execute25'
 def load(kind):
  ref=e[kind];p=Path(ref['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256'];return json.loads(p.read_text())
 r,l,z=load('result'),load('launch'),load('recovery')
 assert e['expected_package']=='gdn_parent_state_no_oracle_v2'
 load('source_pins');load('final_seal')
 assert z['source_pin_file_sha256']==e['source_pins']['sha256']==e['expected_source_pins_sha256']
 assert z['source_pins_verified']==3073 and z['baseline_cache_files_verified']==1033
 assert z['launch_sha256']==e['launch']['sha256'] and not l.get('timeout') and not l.get('error') and l.get('finished')
 assert l['returncode']==z['actual_returncode']==0 and l['child_absent'] and z['child_absent']
 assert z.get('lock_available',z.get('lock_reacquired')) and not z.get('remaining_process_group')
 assert l['result_sha256']==z['result_sha256']==e['result']['sha256']
 from gdn_parent_state_no_oracle_v2.result_gate import audit
 assert audit(r)['passed']
 return e
