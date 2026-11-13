from pathlib import Path
import json
from packed_grouped_qualification_v1.contract import sha
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from selected_ranks_qualification_v1.run import result_gate
def qualification(root):
 p=Path("/home/n/work/TileIR/public-tileir-blas-20261111/selected_ranks_qualification_v1/selectedRanksQual01/RESULT.json");c=Path("/home/n/work/TileIR/public-tileir-blas-20261111/selected_ranks_qualification_v1/selectedRanksQual01-recovery.json");l=Path("/home/n/work/TileIR/public-tileir-blas-20261111/selected_ranks_qualification_v1/selectedRanksQual01-launch.json")
 assert sha(p)=="f7d66a2f0bc62d4979ad71e94d8b25d635f0d51224999e963a3ea0b7e9a57039" and sha(c)=="71f8b67cff95d1da220ffa63779381e8aeb5ab5d80aeffefd99157dbc1220eb6" and sha(l)=="e252729970d03d4484dc0485bb5d2dc370f28a30b162a72de2d177fe88bc4f2c"
 assert result_gate(json.loads(p.read_text()))['raw_pass'] and json.loads(l.read_text())['returncode']==0
 rec=json.loads(c.read_text());assert rec['child_absent'] and not rec['remaining_process_group'] and rec['lock_available']
 return dict(result_sha256=sha(p),scope='Two actual rank banks qualified; all16 model shadows mandatory')
