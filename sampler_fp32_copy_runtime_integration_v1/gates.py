import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from sampler_fp32_copy_qualification_v1.launch import result_gate
def qualification(root):
 base=Path('/home/n/work/TileIR/public-tileir-blas-20261111/sampler_fp32_copy_qualification_v1');p=base/'samplerFP32CopyQual01/RESULT.json';r=base/'samplerFP32CopyQual01-recovery.json'
 assert hashlib.sha256(p.read_bytes()).hexdigest()=='e1e5edec7bc69c4b9fa7d1e3810ed9ceefd26ebb31a8ecb2506ea92b45933aba'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='a38ad4e6b863a5cac14c6e09143b5861133f2d27e33cc4c1f67e80c64bc36eac'
 assert result_gate(json.loads(p.read_text()))['raw_pass'];c=json.loads(r.read_text());assert c['actual_returncode']==0 and c['child_absent'] and c['lock_reacquired'] and not c['remaining_process_group']
 return dict(qualified_actual_banks=2,model_shadow_still_required=True)
