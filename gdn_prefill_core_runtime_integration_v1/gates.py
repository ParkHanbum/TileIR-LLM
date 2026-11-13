from pathlib import Path
import json
from packed_grouped_qualification_v1.contract import sha
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from gdn_prefill_chain_qualification_v1.run import result_gate
def qualification(root):
 b=Path('/home/n/work/TileIR/public-tileir-blas-20261111/gdn_prefill_chain_qualification_v1');p=b/'gdnPrefillChain01/RESULT.json';c=b/'gdnPrefillChain01-recovery.json'
 assert sha(p)=='a572ff31fddeeeec1cd6aa20a1b2e3cc2d24060fe8bb7f5a439c14e80fe05b54' and sha(c)=='66b604dda7589c79ecf0464076ebb680224bd51f88b7e2507a02dd9e43c2127f'
 assert result_gate(json.loads(p.read_text()))['raw_pass'];rec=json.loads(c.read_text());assert rec['child_absent'] and rec['lock_available']
 return dict(result_sha256=sha(p),scope='One zero-initial native2/TileIR2 six-stage bank qualified; fresh24-layer shadow mandatory')
