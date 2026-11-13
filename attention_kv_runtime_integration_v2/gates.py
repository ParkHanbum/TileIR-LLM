from pathlib import Path
import json
from packed_grouped_qualification_v1.contract import sha
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from attention_kv_qualification_v2.run import result_gate
def qualification(root):
 p=Path('/home/n/work/TileIR/public-tileir-blas-20261111/attention_kv_qualification_v2/attentionKVQual01/RESULT.json');c=Path('/home/n/work/TileIR/public-tileir-blas-20261111/attention_kv_qualification_v2/attentionKVQual01-recovery.json')
 assert sha(p)=='08323fc230473925eefd7fffda741210d08e520d53d8999d3cf3bf0d1e37426c' and sha(c)=='b4f3c068621e054277d6942ab78bd422f0aacd47ef154b52d1a241823f7a0abb'
 assert result_gate(json.loads(p.read_text()))['raw_pass']
 rec=json.loads(c.read_text());assert rec['child_pid_absent'] and rec['lock_available']
 return dict(result_sha256=sha(p),scope='Two compact-prefix actual cases qualified; new live128 shadow mandatory')
