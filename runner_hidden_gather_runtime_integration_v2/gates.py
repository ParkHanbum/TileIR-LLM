import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from runner_hidden_gather_qualification_v2.launch import result_gate
def qualification(root):
 base=Path('/home/n/work/TileIR/public-tileir-blas-20261111/runner_hidden_gather_qualification_v2');p=base/'hiddenGatherQual02/RESULT.json';r=base/'hiddenGatherQual02-recovery.json'
 assert hashlib.sha256(p.read_bytes()).hexdigest()=='54c2dbcb7826124a47f6e57a25131335063f370f3b5ae29d851013c3d6475df7'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='a9d4519561b1e8dbfd73cb0f30cf8855213cd99ab60a1ddadfece72619b9af20'
 assert result_gate(json.loads(p.read_text()))['raw_pass'];c=json.loads(r.read_text());assert c['actual_returncode']==0 and c['child_absent'] and c['lock_reacquired'] and not c['remaining_process_group']
 return dict(qualified_actual_banks=2,model_shadow_still_required=True)
