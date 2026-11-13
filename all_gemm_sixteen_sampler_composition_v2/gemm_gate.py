from all_gemm_runtime_integration_v1.contract import plan,coverage
from pathlib import Path
from all_gemm_runtime_integration_v1.metrics import passing

def audit(r, cache_root):
 try:
  p=plan();cache=r['cache_preseed'];assert cache['destination']==str(Path(cache_root)/'cache_v1')and cache['files_verified']and cache['separate_inodes_verified']and cache['file_count']>0
  assert r['complete']and r['engine_shutdown']and not r.get('error')and not r.get('cleanup_errors')
  assert r['adapter_restore'][0]['restored']and not r['adapter_restore'][0].get('conflicts')
  install=r['adapter_install'][0];assert install['selected_modules']==153 and install['excluded_guarded_modules']==25
  assert {x['path']for x in install['modules']}=={x['path']for x in p['selected']}
  policy=r['precision_policy'];assert policy['policy']=='unchanged native'and policy['initial']==dict(bf16=True,split_k=True)and policy['restored']and policy['restored_split_k']
  a,b=r['shadow_request'],r['replacement_request'];shadow={}
  for arm,native in ((a,2448),(b,0)):
   d=arm['selected_dispatch'];assert d['shadow_pass']and not d['failed']and not d['pending']and d['excluded_apply_calls']==d['selected_vendor_fallback_calls']==0
   assert d['selected_calls']==d['tileir_calls']==2448 and d['native_calls']==native and d['k128_calls']==360 and d['existing_calls']==2088 and coverage(d['rows'],p)
   for row in d['rows']:
    assert row['passed']and row['weight_metadata_match']and row['input_preserved']and row['output_contract_pass']and row['tileir_calls']==1 and row['native_calls']==(1 if native else 0)
    assert row['candidate_backend']==('K128'if row['kind']=='ba'and row['phase']=='decode'else'existing')
    key=(row['path'],row['call_index'])
    if native:
     c=row['comparison'];assert passing(c)and c['elements']==row['logical_MNK'][0]*row['logical_MNK'][1];assert c['native_sha256']==c['tile_sha256']==row['native_output_sha256']==row['output_sha256'];shadow[key]=row
    else:
     old=shadow[key];assert row['same_shadow_inputs']and row['same_shadow_output']and row['output_finite']
     assert row['input_sha256']==old['input_sha256']and row['weight_key']==old['weight_key']and row['output_sha256']==old['native_output_sha256']
   assert arm['cross_runtime_tokens_equal']is True and arm['allow_bf16_reduced_precision_reduction']is True and arm['allow_bf16_reduced_precision_reduction_split_k']is True
  weights=b['selected_dispatch']['weight_boundaries'];tags=['shadow_start','shadow_end','replace_start','replace_end'];assert all(t in weights for t in tags)
  first=weights[tags[0]];assert 1<=first['unique_weights']<=153 and len(first['records'])==first['unique_weights']
  owners=[n for v in first['records'].values()for n in v['owners']];assert len(owners)==153 and set(owners)=={s['path']for s in p['selected']}
  baseline={k:v['sha256']for k,v in first['records'].items()}
  for tag in tags:
   boundary=weights[tag];assert boundary['unique_weights']==len(baseline) and set(boundary['records'])==set(baseline)
   assert boundary['full_bytes']==sum(v['bytes']for v in boundary['records'].values())
   for key,v in boundary['records'].items():assert v['finite']and v['sha256']==baseline[key]and v['owners']==first['records'][key]['owners']and v['metadata']==first['records'][key]['metadata']and v['bytes']>0
  assert a['output_token_ids']==b['output_token_ids']and len(a['output_token_ids'])==32 and all(len(x)==16 for x in a['output_token_ids'])
  g=r['public_logprob_gate'];assert g['passed']and g['exact_numeric']and g['max_abs']==0 and g['values']==2560
  assert not r.get('cycles')and not r.get('warmup_records')
  return dict(passed=True,errors=[],scope='GEMM projection:153 owners/2448calls/13contracts; other components audited separately; W phase-boundary bytes only; no performance claim')
 except (KeyError,AssertionError,TypeError,IndexError):return dict(passed=False,errors=['Incomplete all-GEMM native-shadow/replacement proof'])
