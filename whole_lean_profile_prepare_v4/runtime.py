"""Post-request CPU analysis; no framework imports or artifact discovery."""
import json
from .analyze import analyze

def summarize_trace(path,mode):
    events=json.loads(path.read_text())['traceEvents']
    # No generated/loaded provenance invented at runtime. Exact evidence joins
    # happen after coordinator recovery in a separately sealed analysis result.
    result=analyze(events,mode)
    output=path.with_suffix('.census.json')
    output.write_text(json.dumps(result,indent=2)+'\n')
    return dict(denominator_ready=result['denominator_ready'],compute_denominator_count=result['compute_denominator_count'],compute_count=result['compute_count'],compute_duration_us=result['compute_duration_us'],loaded_artifact_verified=result['loaded_artifact_verified'],loaded_artifact_all_verified=result['loaded_artifact_all_verified'],trace_path=str(path),census_path=str(output),timing_enabled=False,loaded_symbol_join_pending=True,all_gpu_work_converted_claim=False,all_vllm_apis_converted_claim=False)
