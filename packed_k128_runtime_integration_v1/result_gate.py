from .adapter import LAYERS,census

def audit(r):
 try:
  assert r['complete'] and r['engine_shutdown'] and not r.get('error') and not r.get('cleanup_errors')
  assert r['adapter_restore'][0]['restored']and r['adapter_restore'][0]['other_apply_bindings_unchanged']
  p=r['precision_policy'];assert p['restored']and p['restored_split_k']and p['policy']=='unchanged native'and p['initial']==dict(bf16=True,split_k=True)
  a,b=r['shadow_request'],r['replacement_request'];names=[x['path']for x in r['adapter_install'][0]['modules']]
  assert len(names)==24 and len(set(names))==24 and sorted(x['layer']for x in r['adapter_install'][0]['modules'])==list(LAYERS)
  for arm,native in ((a,360),(b,0)):
   d=arm['selected_dispatch'];assert d['shadow_pass']and d['supported_calls']==d['tileir_gate_calls']==360 and d['native_gate_calls']==native and d['prefill_native_calls']==24
   assert d['supported_vendor_fallback_calls']==d['unsupported_native_fallback_calls']==0 and d['other_apply_bindings_unchanged']and d['weights_unchanged']and not d['first_mismatch']and not d['pending']
   assert len(d['rows'])==384 and census(d['rows'],names)
   for row in d['rows']:
    assert row['passed']
    if row['phase']=='decode':
     assert row['candidate_path']=='bf16_k128'and row['tileir_calls']==1 and row['original_calls']==(1 if native else 0)
     assert row['inputs_preserved']and row['input_backing_preserved']and row['output_contract_pass']and row['output_finite']
     if native:
      c=row['comparison'];assert c['raw_equal']and c['raw_mismatches']==c['strict_failures']==c['candidate_nonfinite']==c['reference_nonfinite']==0 and c['elements']==2048
     else:assert row['same_shadow_inputs']and row['same_shadow_output']
    else:assert row['original_calls']==1 and row['tileir_calls']==0
   assert arm['cross_runtime_tokens_equal']is True and arm['allow_bf16_reduced_precision_reduction']is True and arm['allow_bf16_reduced_precision_reduction_split_k']is True
  assert a['output_token_ids']==b['output_token_ids']and len(a['output_token_ids'])==32 and all(len(x)==16 for x in a['output_token_ids'])
  g=r['public_logprob_gate'];assert g['passed']and g['exact_numeric']and g['max_abs']==0 and g['values']==2560
  assert not r.get('cycles')and not r.get('warmup_records')
  return dict(passed=True,errors=[],scope='360 ba decode calls only; other GEMMs native; no performance proof')
 except (KeyError,AssertionError,TypeError,IndexError):return dict(passed=False,errors=['Incomplete ba decode K128 shadow/replacement proof'])
