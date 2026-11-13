import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 p=P/'paged_attention_qualification_v8';r=p/'pagedAttentionQual08/RESULT.json';c=p/'pagedAttentionQual08-recovery.json';l=p/'pagedAttentionQual08-launch.json'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='41b2b9a5ee3e534df513cd72cd0c7779104a728ba05ac02cce95735edbb6c14f'
 assert hashlib.sha256(c.read_bytes()).hexdigest()=='efc35f197e2292e9246bfe4b5c47f99f30b73bad928f7aa99bf0050160863173'
 from paged_attention_qualification_v8.run import result_gate
 assert result_gate(json.loads(r.read_text()))['raw_pass'] and json.loads(l.read_text())['returncode']==0
 rec=json.loads(c.read_text());assert rec.get('child_absent',rec.get('child_pid_absent')) and rec['lock_available']
 return dict(scope='Three isolated banks: layer7 prefill53 + layer3 prefill53/decode54; 96 live calls remain unqualified',result_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
