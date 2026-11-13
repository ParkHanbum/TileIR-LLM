import json,hashlib
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from indexed_decode_conv_qualification_v1.run import result_gate
def qualification(root):
 e=json.loads((Path(root)/'QUALIFICATION-RECEIPT.json').read_text())
 for p,h in e['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 r=json.loads(Path(e['result']).read_text());l=json.loads(Path(e['launch']).read_text());c=json.loads(Path(e['recovery']).read_text())
 assert result_gate(r)['raw_pass'] and l['returncode']==0 and c['child_pid_absent'] and c['lock_available'];return e
