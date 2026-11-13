import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 p=P/'sample_counts_qualification_v1';r=p/'sampleCountsQual01/RESULT.json';c=p/'sampleCountsQual01-recovery.json';l=p/'sampleCountsQual01-launch.json'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='9f4e229e3dce7a7172825196a78716e2419b7bac477d305290e279658b2401f3'
 assert hashlib.sha256(c.read_bytes()).hexdigest()=='b0921146934af2ef9ffc366a8d3549b57cc4c26a35163e758615b3816a607b35'
 assert json.loads(r.read_text())['gate']['raw_pass'] and json.loads(l.read_text())['returncode']==0
 rec=json.loads(c.read_text());assert rec['child_pid_absent'] and rec['lock_available']
 return dict(scope='Two frozen actual cases mapped-UVA sample counts qualified; all16 native shadow calls still required',result_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
