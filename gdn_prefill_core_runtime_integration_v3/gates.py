from pathlib import Path
import json
from packed_grouped_qualification_v1.contract import sha
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from gdn_prefill_chain_qualification_v1.run import result_gate
def qualification(root):
 b=Path('/home/n/work/TileIR/public-tileir-blas-20261111/gdn_prefill_chain_qualification_v1');p=b/'gdnPrefillChain01/RESULT.json';c=b/'gdnPrefillChain01-recovery.json'
 assert sha(p)=='a572ff31fddeeeec1cd6aa20a1b2e3cc2d24060fe8bb7f5a439c14e80fe05b54' and sha(c)=='66b604dda7589c79ecf0464076ebb680224bd51f88b7e2507a02dd9e43c2127f'
 assert result_gate(json.loads(p.read_text()))['raw_pass'];rec=json.loads(c.read_text());assert rec['child_absent'] and rec['lock_available']
 q=Path('/home/n/work/TileIR/public-tileir-blas-20261111/gdn_solve_recurrence_qualification_v6');rp=q/'gdnSolveRecurrence06/RESULT.json';rc=q/'gdnSolveRecurrence06-recovery.json'
 assert sha(rp)=='f45e0b5388d7641a561244f295913ef1d95e1ad983c78a4f13270555a56c5920' and sha(rc)=='e45a2f261f24857254833eafb66d4994aa4be8d64ccdf48dd2c2b20b954fbc6f'
 from gdn_solve_recurrence_qualification_v6.run import result_gate as solve_gate
 assert solve_gate(json.loads(rp.read_text()))['raw_pass'];rr=json.loads(rc.read_text());assert rr['child_absent'] and rr['lock_available']
 return dict(solve_v6_result_sha=sha(rp),result_sha256=sha(p),scope='Historical v5 six-stage bank plus v6 solve two-bank pass; new24-layer v6 chain shadow mandatory')
