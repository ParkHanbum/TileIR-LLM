import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 p=P/'silu_runtime_qualification_v1';r=p/'silu01/RESULT.json';c=p/'silu01-recovery.json';l=p/'silu01-launch.json'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='9141663fa0326a4ad81b7a29531218e0720c724921803bd46a96d748d952c64a'
 assert hashlib.sha256(c.read_bytes()).hexdigest()=='caa018a968aa79d66be46d05b23f0a8aee5f69ee89801e8884168d26f7644317'
 assert json.loads(r.read_text())['gate']['raw_pass'] and json.loads(l.read_text())['returncode']==0
 rec=json.loads(c.read_text());assert rec['child_pid_absent'] and rec['lock_available']
 return dict(scope='Two frozen actual cases MLP activation inputs qualified; all512 live calls still required',result_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
