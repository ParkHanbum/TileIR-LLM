from pathlib import Path
import json
from packed_grouped_qualification_v1.contract import sha
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from greedy_sampler_qualification_v1.run import result_gate
def qualification(root):
 p=Path("/home/n/work/TileIR/public-tileir-blas-20261111/greedy_sampler_qualification_v1/greedySamplerQual01/RESULT.json");c=Path("/home/n/work/TileIR/public-tileir-blas-20261111/greedy_sampler_qualification_v1/greedySamplerQual01-recovery.json");l=Path("/home/n/work/TileIR/public-tileir-blas-20261111/greedy_sampler_qualification_v1/greedySamplerQual01-launch.json")
 assert sha(p)=="775520bb74c375e03071566bd3730a324d2938060a01328f94c79d02b29d2af7" and sha(c)=="0f0d9a79d69360b52a51a38f807406be0c1152dd47b81fe73e38c3cb631ddab4" and sha(l)=="b3eeb1e9f6e74617ebf5a21c7635ae798df0ba3fd205384ea6a373d9fb84352b"
 assert result_gate(json.loads(p.read_text()))['raw_pass'] and json.loads(l.read_text())['returncode']==0
 rec=json.loads(c.read_text());assert rec.get('child_absent',rec.get('child_pid_absent')) and rec['lock_available']
 return dict(result_sha256=sha(p),scope='Two actual greedy banks only; all16 live native shadows mandatory')
