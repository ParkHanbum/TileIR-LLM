"""CPU-only exact71-case requirements, including independent result audit."""
from collections import Counter
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
CORE_PINS_SHA256='6be9fe49d9f2a97c76a203a5e6f379f5e81d7e4a07378956e8b46cdf8389a9c9'


def expected_case_keys():
    v1=json.loads((ROOT/'OBSERVED-CAT-CONTRACTS.json').read_text())
    v2=json.loads((ROOT/'OBSERVED-TOKEN-CAT-v2.json').read_text())
    keys=[(site,i,'baseline_raw') for site,r in v1['sites'].items() for i in range(len(r['variants']))]
    keys.extend((site,0,'distinct_source_identity') for site in ('cat_gdn_decode','cat_gdn_prefill'))
    keys.extend(('cat_input_ids',i,'baseline_raw') for i in range(len(v2['variants'])))
    if len(keys)!=71 or len(set(keys))!=71:raise RuntimeError('expected71 unique raw cases')
    return keys


def exact_counters():
    return dict(calls=1,allocations=1,allocation_failures=0,native_fallbacks=0,
        native_failures=0,launch_attempts=1,launch_failures=0,tileir_cats=1,fallback_reasons={})


def counters_match(counts):
    expected=exact_counters()
    return counts==expected and all(type(counts.get(k)) is int for k in expected if k!='fallback_reasons')


def audit_result(result):
    errors=[];cases=result.get('cases',[])
    expected=expected_case_keys()
    actual=[(x.get('site'),x.get('variant'),x.get('pattern','baseline_raw')) for x in cases]
    if Counter(actual)!=Counter(expected):errors.append('missing_duplicate_or_unexpected_cases')
    if result.get('expected_cases')!=[list(x) for x in expected] or result.get('expected_case_count')!=71:
        errors.append('expected_case_manifest_mismatch')
    if result.get('schema')!='cat-v3-gpu-raw-diagnostic' or result.get('scope')!='all':errors.append('schema_or_scope')
    if result.get('core_manifest_sha256')!=CORE_PINS_SHA256:errors.append('core_manifest_mismatch')
    if result.get('complete') is not True:errors.append('child_incomplete')
    if result.get('timing_executed') is not False or result.get('model_executed') is not False:errors.append('unexpected_timing_or_model')
    if result.get('error'):errors.append('child_error')
    required=('passed','exact_numpy','exact_native','inputs_unchanged','input_metadata_unchanged',
              'custom_output_contract','native_output_contract','outputs_mutually_disjoint','full_counters_exact')
    for i,case in enumerate(cases):
        for flag in required:
            if case.get(flag) is not True:errors.append(f'case{i}_{flag}')
        if not counters_match(case.get('counters',{})):errors.append(f'case{i}_full_counters')
    return dict(passed=not errors,expected_cases=71,actual_cases=len(cases),errors=errors)
