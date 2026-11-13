import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 p=P/'paged_attention_qualification_v10';r=p/'pagedAttentionQual10/RESULT.json';c=p/'pagedAttentionQual10-recovery.json';l=p/'pagedAttentionQual10-launch.json'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='1b19536572817796cc3bc25658f5c8b0d2f1b461870d2b385056dbc79146e078'
 assert hashlib.sha256(c.read_bytes()).hexdigest()=='99f4e305ee7a3b42fd4477c8ea4e832c53d2ec77f7bac35ee416dfd3cac8e2a0'
 from paged_attention_qualification_v10.run import result_gate
 assert result_gate(json.loads(r.read_text()))['raw_pass'] and json.loads(l.read_text())['returncode']==0
 rec=json.loads(c.read_text());assert rec['child_absent'] is True and rec['lock_reacquired'] is True and not rec['remaining_process_group'] and rec['actual_returncode']==0
 return dict(scope='Six isolated banks including layer3 decode65/68; full128 live calls remain unqualified',result_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
