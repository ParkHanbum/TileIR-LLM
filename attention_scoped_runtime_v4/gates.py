import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 p=P/'paged_attention_qualification_v9';r=p/'pagedAttentionQual09/RESULT.json';c=p/'pagedAttentionQual09-recovery.json';l=p/'pagedAttentionQual09-launch.json'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='7700a2bce62e09eb25025376858ed438c80a2e2caca6465c94398e100cb90632'
 assert hashlib.sha256(c.read_bytes()).hexdigest()=='6daec14150601f3d9541398cd5b8c7defdf6a951463cf8f367062039b9c6d549'
 from paged_attention_qualification_v9.run import result_gate
 assert result_gate(json.loads(r.read_text()))['raw_pass'] and json.loads(l.read_text())['returncode']==0
 rec=json.loads(c.read_text());assert rec['child_absent'] is True and rec['lock_reacquired'] is True and not rec['remaining_process_group'] and rec['actual_returncode']==0
 return dict(scope='Four isolated banks: layer11/layer7 prefill53 + layer3 prefill53/decode54; 96 live calls remain unqualified',result_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
