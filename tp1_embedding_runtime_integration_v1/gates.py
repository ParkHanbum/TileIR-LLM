import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 p=P/'tp1_embedding_qualification_v1';r=p/'tp1EmbeddingQual01/RESULT.json';c=p/'tp1EmbeddingQual01-recovery.json';l=p/'tp1EmbeddingQual01-launch.json'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='ae954d753ea74213fabdabb2da852e3bdc9f5c44e29c8d5be84c4693094df9d3'
 assert hashlib.sha256(c.read_bytes()).hexdigest()=='27a820c51009acc0c31a71cc52c5bbab455dd48fc73f0670ff3f5f3b43cfbe15'
 from tp1_embedding_qualification_v1.launch import result_gate
 assert result_gate(json.loads(r.read_text()))['raw_pass'] and json.loads(l.read_text())['returncode']==0
 rec=json.loads(c.read_text());assert rec['child_pid_absent'] and rec['lock_available']
 return dict(scope='Derived table isolated two phases qualified; all16 actual live table calls remain unqualified',result_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
