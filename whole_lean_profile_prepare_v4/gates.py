"""CPU gates for exact v7 lifecycle plus request-correlated profiling."""
import json
from pathlib import Path
from .admission import require_ready
from .analyze import analyze

def qualification(root):
    assert Path(root).resolve()==Path(__file__).resolve().parent
    return require_ready()

def audit(result,root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parent
    from whole_lean_extensions_runtime_v7.gates import audit as original
    base=original(result,root) # profile root required by inherited destination-bound cache gate
    if not base['passed']:return base
    try:
        for mode in ('native','lean'):
            item=result[mode+'_request'];summary=item['profile_partition']
            assert item['profiled'] is True and item['whole_ms'] is None
            assert summary['timing_enabled'] is False and summary['loaded_symbol_join_pending'] is True
            path=Path(summary['trace_path']);census=Path(summary['census_path'])
            assert path.name==mode+'.trace.json' and census==path.with_suffix('.census.json')
            assert path.parent.parent==root and census.parent==path.parent
            # Recompute correlation from the actual saved trace, not result counters.
            checked=analyze(json.loads(path.read_text())['traceEvents'],mode)
            assert checked['denominator_ready'] is True
            assert checked['compute_denominator_count']==summary['compute_denominator_count']
            saved=json.loads(census.read_text());assert saved==checked
            assert summary['compute_count']==checked['compute_count']
            assert summary['compute_duration_us']==checked['compute_duration_us']
            assert summary['loaded_artifact_verified']==checked['loaded_artifact_verified']
            assert summary['loaded_artifact_all_verified']==checked['loaded_artifact_all_verified']
            assert not summary['all_gpu_work_converted_claim'] and not summary['all_vllm_apis_converted_claim']
        return dict(passed=True,profile_only=True,performance=False,physical_tileir_numerator_pending=True,scope='Exact v7 endpoint/census and same-request launch-correlated compute denominator; transport/unknown retained separately')
    except (AssertionError,KeyError,TypeError,ValueError,OSError):
        return dict(passed=False,scope='profile_trace_or_lifecycle')
