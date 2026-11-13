from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
def qualification(root):
 from pathlib import Path
 import json,hashlib
 p=Path(root).parent/'gdn_parent_state_synthetic_v2/gdnParentStateSynthetic02/RESULT.json'
 assert hashlib.sha256(p.read_bytes()).hexdigest()=='ce88fc3f48a237c739d121606761bc1231093e3daa20491c97741cd387c5931d'
 r=json.loads(p.read_text());assert r['complete'] and all(x['passed'] for x in r['records'])
 return dict(synthetic_pass=True,actual_model_parent_pending=True)
