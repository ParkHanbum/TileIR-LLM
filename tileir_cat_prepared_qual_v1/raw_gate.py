"""Reuse every strict v3 raw gate and additionally identify this candidate."""
from tileir_cat_v1.raw_gate_v2 import (
    CORE_PINS_SHA256, expected_case_keys, exact_counters, counters_match,
    audit_result as _original_audit,
)
CANDIDATE_PINS_SHA256='05990c40e6eb102572556ec8cc441c0cde3985d0d94db8b9012a1d0bd3b21fef'
CANDIDATE_API='tileir_cat_prepared_v1.cat_observed_prepared'


def audit_result(result):
    audit=_original_audit(result)
    if result.get('candidate_manifest_sha256')!=CANDIDATE_PINS_SHA256:
        audit['errors'].append('candidate_manifest_mismatch')
    if result.get('candidate_api')!=CANDIDATE_API:
        audit['errors'].append('candidate_api_mismatch')
    audit['passed']=not audit['errors']
    return audit
