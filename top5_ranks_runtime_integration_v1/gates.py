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
 top=Path('/home/n/work/TileIR/public-tileir-blas-20261111/top5_qualification_v2/top5Qual02/RESULT.json');toprec=Path('/home/n/work/TileIR/public-tileir-blas-20261111/top5_qualification_v2/top5Qual02-recovery.json');toplaunch=Path('/home/n/work/TileIR/public-tileir-blas-20261111/top5_qualification_v2/top5Qual02-launch.json')
 assert sha(top)=='977222f1ddab72fe23fe434c2d0adf0ff871a2120b290297ed6e891b0d7c2b54' and sha(toprec)=='9b9155f2219f6408eb34032716e65d4cc378179c4509afa2b9e5b598d9dcc840' and sha(toplaunch)=='18c2fafe4c9056b12a192912b609d60ad9197a759213c0b08e52d42d3bde990b'
 from top5_qualification_v2.run import result_gate as top_gate
 assert top_gate(json.loads(top.read_text()))['raw_pass'] and json.loads(toplaunch.read_text())['returncode']==0
 rr=json.loads(toprec.read_text());assert rr['child_absent'] and not rr['remaining_process_group'] and rr['lock_available']
 return dict(rank_result_sha256=sha(p),top5_result_sha256=sha(top),scope='Two ranks banks and three top5 banks; full16 model raw shadow mandatory; scores native')
