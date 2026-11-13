import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 p=P/'paged_attention_qualification_v7';r=p/'pagedAttentionQual07/RESULT.json';c=p/'pagedAttentionQual07-recovery.json';l=p/'pagedAttentionQual07-launch.json'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='c7650ae8bff6706b9bf57b2d4684ad5347d65e0e5c683cb283bd7eec08e8c622'
 assert hashlib.sha256(c.read_bytes()).hexdigest()=='b6056a1c88e5a523c3ec9018e6ac700dc302d46bd6be6c7cbe6c0f34ff974b71'
 from paged_attention_qualification_v7.run import result_gate
 assert result_gate(json.loads(r.read_text()))['raw_pass'] and json.loads(l.read_text())['returncode']==0
 rec=json.loads(c.read_text());assert rec.get('child_absent',rec.get('child_pid_absent')) and rec['lock_available']
 return dict(scope='Two isolated actual lengths53/54 only; 96 live calls remain unqualified',result_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
