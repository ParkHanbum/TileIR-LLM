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
 return dict(isolated_result_sha256='f6278c9f0c8d8a30c594e4a69809b89687416afc5c066dd4a9075b1c264c5aad',scope='Two exact constructor contracts; model shadow still required')
