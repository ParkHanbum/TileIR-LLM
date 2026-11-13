from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from pathlib import Path
import json,hashlib
def metadata_qualification(root):
 p=Path(root)/'OBSERVED.json';d=json.loads(p.read_text())
 for f,h in d['source_files'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 assert d['prefill']['calls']==d['prefill']['returns']==1 and d['prefill']['errors']==0
 links=json.loads((Path(root)/'PREREQUISITE.json').read_text())
 for f,h in links.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 base=Path(root).parent/'lean25_metadata_inventory_v2'
 result=json.loads((base/'leanMetadata02/RESULT.json').read_text());launch=json.loads((base/'leanMetadata02-launch.json').read_text());recovery=json.loads((base/'leanMetadata02-recovery.json').read_text());pins=json.loads((base/'PINS.json').read_text())
 assert result['complete'] is True and type(launch['returncode']) is int and launch['returncode']==0 and launch.get('finished')
 assert type(recovery['actual_returncode']) is int and recovery['actual_returncode']==0 and recovery['child_absent'] is True and recovery['remaining_process_group']==[] and recovery['lock_available'] is True
 assert recovery['result_sha256']==hashlib.sha256((base/'leanMetadata02/RESULT.json').read_bytes()).hexdigest() and recovery['launch_sha256']==hashlib.sha256((base/'leanMetadata02-launch.json').read_bytes()).hexdigest()
 assert recovery['pins_checked']==len(pins)
 for f,h in pins.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 return dict(metadata_bound=True,metadata_exit0=True,independent_recovery=True,actual_payload_validation_pending=True,shadow_only=True)

def qualification(root):
 metadata_qualification(root)
 link=json.loads((Path(root)/'SHADOW-PROOF.json').read_text())
 for f,h in link.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 base=Path(root).parent/'prepare_prefill_inputs_runtime_shadow_v1';name='preparePrefillShadow01'
 result=json.loads((base/name/'RESULT.json').read_text());launch=json.loads((base/(name+'-launch.json')).read_text());recovery=json.loads((base/(name+'-recovery.json')).read_text());pins=json.loads((base/'PINS.json').read_text())
 from prepare_prefill_inputs_runtime_shadow_v1.result_gate import audit
 assert audit(result)['passed'] and type(launch['returncode'])is int and launch['returncode']==0 and launch['finished']
 assert type(recovery['actual_returncode'])is int and recovery['actual_returncode']==0 and recovery['child_absent'] is True and recovery['remaining_process_group']==[] and recovery['lock_available'] is True
 assert recovery['result_sha256']==hashlib.sha256((base/name/'RESULT.json').read_bytes()).hexdigest() and recovery['launch_sha256']==hashlib.sha256((base/(name+'-launch.json')).read_bytes()).hexdigest()
 assert recovery['pins_checked']==len(pins)
 for f,h in pins.items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==h
 return dict(actual_shadow_pass=True,actual_exit0=True,recovery=True,actual_replacement_pending=True)
