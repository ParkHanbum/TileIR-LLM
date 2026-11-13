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
 score=Path('/home/n/work/TileIR/public-tileir-blas-20261111/selected_logprobs_qualification_v1/selectedLogprobsQual01/RESULT.json');recovery=Path('/home/n/work/TileIR/public-tileir-blas-20261111/selected_logprobs_qualification_v1/selectedLogprobsQual01-recovery.json');launch=Path('/home/n/work/TileIR/public-tileir-blas-20261111/selected_logprobs_qualification_v1/selectedLogprobsQual01-launch.json')
 assert sha(score)=='41e89640ae8be6715334b17048749c1824f67f0665ca8fb5f26b7d67f4fcde60' and sha(recovery)=='c973d6386253db35e2880984237854a38a329a07532452a0070165efa1d7d1ef' and sha(launch)=='6eeae65dd5ddef533b793eea215b340d06888d0a625293481e35d8c6ba1c1043'
 from selected_logprobs_qualification_v1.run import result_gate as scores_gate
 assert scores_gate(json.loads(score.read_text()))['raw_pass'] and json.loads(launch.read_text())['returncode']==0
 rec2=json.loads(recovery.read_text());assert rec2['child_absent'] and not rec2['remaining_process_group'] and rec2['lock_available']
 return dict(result_sha256=sha(p),scores_sha256=sha(score),scope='Two actual rank banks qualified; all16 model shadows mandatory')
