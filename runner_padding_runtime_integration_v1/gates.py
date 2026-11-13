import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from runner_constructors_qualification_v1.run import result_gate
def qualification(root):
 p=Path('/home/n/work/TileIR/public-tileir-blas-20261111/runner_constructors_qualification_v1/runnerConstructorsQual01/RESULT.json');r=Path('/home/n/work/TileIR/public-tileir-blas-20261111/runner_constructors_qualification_v1/runnerConstructorsQual01-recovery.json')
 assert hashlib.sha256(p.read_bytes()).hexdigest()=='f6278c9f0c8d8a30c594e4a69809b89687416afc5c066dd4a9075b1c264c5aad'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='93dbe45b47438e84de66533dde1b809a6c21ff39fe11085e2fc34d3db77e217f'
 assert result_gate(json.loads(p.read_text()))['raw_pass']
 c=json.loads(r.read_text());assert c['actual_returncode']==0 and c['child_absent'] and not c['remaining_process_group'] and c['lock_available']
 q=Path('/home/n/work/TileIR/public-tileir-blas-20261111/runner_bool_fill_qualification_v1/runnerBoolFillQual01/RESULT.json');qr=Path('/home/n/work/TileIR/public-tileir-blas-20261111/runner_bool_fill_qualification_v1/runnerBoolFillQual01-recovery.json')
 assert hashlib.sha256(q.read_bytes()).hexdigest()=='6ba14880167ee507d917017391133f90a64650a45a487d928c004e81ba7ac8e4' and hashlib.sha256(qr.read_bytes()).hexdigest()=='15b02a7a76e61b9f5d2ba8f378b46573b29ebe00b8c4df606061d4dde7b2ef31'
 from runner_bool_fill_qualification_v1.run import result_gate as fill_gate
 assert fill_gate(json.loads(q.read_text()))['raw_pass']
 rr=json.loads(qr.read_text());assert rr['actual_returncode']==0 and rr['child_absent'] and not rr['remaining_process_group'] and rr['lock_reacquired']
 return dict(isolated_fill_sha256='6ba14880167ee507d917017391133f90a64650a45a487d928c004e81ba7ac8e4',isolated_result_sha256='f6278c9f0c8d8a30c594e4a69809b89687416afc5c066dd4a9075b1c264c5aad',scope='Two exact constructor contracts; model shadow still required')
