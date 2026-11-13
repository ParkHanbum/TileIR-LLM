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
 ordered=Path('/home/n/work/TileIR/public-tileir-blas-20261111/selected_logprobs_ordered_qualification_v1/selectedLogprobsOrdered01/RESULT.json');ordered_rec=Path('/home/n/work/TileIR/public-tileir-blas-20261111/selected_logprobs_ordered_qualification_v1/selectedLogprobsOrdered01-recovery.json');ordered_launch=Path('/home/n/work/TileIR/public-tileir-blas-20261111/selected_logprobs_ordered_qualification_v1/selectedLogprobsOrdered01-launch.json')
 assert sha(ordered)=='ba0169604ba219dd2a5b1f8f048cacca92ef56977efff8ed60996d531931e691' and sha(ordered_rec)=='9ff8c972e79abff28dd005bb0e671e62ae73e103163bbb2afd911cc621c8e532' and sha(ordered_launch)=='7fa1f15fcde426b75cc3cf3c993121ff6e265d72c23c1c9056771c052dc5bc70'
 from selected_logprobs_ordered_qualification_v1.run import result_gate as ordered_gate
 assert ordered_gate(json.loads(ordered.read_text()))['raw_pass'] and json.loads(ordered_launch.read_text())['returncode']==0
 recovery=json.loads(ordered_rec.read_text());assert recovery['child_absent'] and not recovery['remaining_process_group'] and recovery['lock_available']
 prior=Path('/home/n/work/TileIR/public-tileir-blas-20261111/top5_ranks_runtime_integration_v1/top5RanksIntegration01/RESULT.json');prior_rec=Path('/home/n/work/TileIR/public-tileir-blas-20261111/top5_ranks_runtime_integration_v1/top5RanksIntegration01-recovery.json')
 assert sha(prior)=='78a05b937185561be686742d059e6a53c51e19b9629ebfa1435de7be6ae91a6d' and sha(prior_rec)=='452dd78630dab66a822b404661fc7a3086ae26ef224005280824ea4643acefcf'
 from top5_ranks_runtime_integration_v1.result_gate import audit as prior_gate
 assert prior_gate(json.loads(prior.read_text()))['passed']
 pr=json.loads(prior_rec.read_text());assert pr['actual_returncode']==0 and pr['child_absent'] and not pr['remaining_process_group'] and pr['lock_available']
 return dict(rank_result_sha256=sha(p),top5_result_sha256=sha(top),ordered_logprob_result_sha256=sha(ordered),scope='Isolated rank/top5/three ordered-score banks; all16 three-way shadow mandatory')
