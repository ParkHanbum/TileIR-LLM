import json,hashlib
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from gdn_decode_output_order_qualification_v1.run import result_gate
def qualification(root):
 e=json.loads((Path(root)/'QUALIFICATION-RECEIPT.json').read_text())
 for p,h in e['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 r=json.loads(Path(e['result']).read_text());l=json.loads(Path(e['launch']).read_text());c=json.loads(Path(e['recovery']).read_text())
 assert result_gate(r)['raw_pass'] and l['returncode']==0 and c['child_absent'] and c['lock_available'] and c['actual_returncode']==0 and not c['remaining_process_group'] and c['source_pins_unchanged'] and c['manifest_unchanged'];return e
