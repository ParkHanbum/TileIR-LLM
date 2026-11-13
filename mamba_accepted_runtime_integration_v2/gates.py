from pathlib import Path
import json,hashlib
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
def qualification(root):
 from mamba_accepted_scatter_qualification_v1.run import result_gate
 p=Path('/home/n/work/TileIR/public-tileir-blas-20261111/mamba_accepted_scatter_qualification_v1/mambaAcceptedScatterQual01/RESULT.json');c=Path('/home/n/work/TileIR/public-tileir-blas-20261111/mamba_accepted_scatter_qualification_v1/mambaAcceptedScatterQual01-recovery.json')
 assert hashlib.sha256(p.read_bytes()).hexdigest()=='fd654d3eb6b80c392c2acee178af1474154e3f4a3e3067fe52733392955b2a51' and hashlib.sha256(c.read_bytes()).hexdigest()=='ab4e9876787407bf9f6c02fcc031eb70364a98794c912fe90c9cf33e1258bffc'
 assert result_gate(json.loads(p.read_text()))['raw_pass'];r=json.loads(c.read_text());assert r['actual_returncode']==0 and r['child_absent'] and not r['remaining_process_group'] and r['lock_reacquired']
 return dict(isolated_result_sha256='fd654d3eb6b80c392c2acee178af1474154e3f4a3e3067fe52733392955b2a51',scope='Actual16banks and2synthetic; full model shadow mandatory')
