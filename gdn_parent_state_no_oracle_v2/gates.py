from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from pathlib import Path
import json,hashlib
from gdn_parent_state_runtime_integration_v2.result_gate import audit

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_proof(result,launch,recovery):
 assert audit(result)['passed'] is True
 for obj,key in ((launch,'returncode'),(recovery,'actual_returncode')):
  assert type(obj.get(key)) is int and obj[key]==0
  assert obj.get('child_absent') is True and obj.get('remaining_process_group')==[] and obj.get('lock_reacquired') is True
  assert not obj.get('error') and not obj.get('timeout')
 assert launch.get('finished') is not None
 assert isinstance(recovery.get('post'),dict)
 return True

def parent_qualification(root):
 # Written and frozen only AFTER the ongoing actual proof has completed/recovered.
 receipt=Path(root)/'PROOF-PREREQUISITE.json'
 assert receipt.is_file(),'Prior actual proof receipt is pending; no execution permitted'
 link=json.loads(receipt.read_text());base=Path(root).parent/'gdn_parent_state_runtime_integration_v2'
 expected=[base/'gdnParentStateIntegration02/RESULT.json',base/'gdnParentStateIntegration02-launch.json',base/'gdnParentStateIntegration02-recovery.json',base/'PINS.json']
 assert link['paths']==[str(p)for p in expected]
 for p in expected:assert sha(p)==link['sha256'][str(p)]
 result,launch,recovery,pins=[json.loads(p.read_text()) for p in expected]
 validate_proof(result,launch,recovery)
 assert launch['pins']==pins
 for p,h in pins.items():assert sha(p)==h,p
 assert launch['result_sha256']==recovery['result_sha256']==sha(expected[0])
 assert recovery['launch_sha256']==sha(expected[1])
 assert recovery['source_pin_file_sha256']==sha(expected[3]) and recovery['source_pins_verified']==len(pins)
 return dict(prior_actual_parent_proof=True,actual_exit0=True,recovery=True,receipt_sha=sha(receipt),target_oracles=0,native_invert_retained=True)

def qualification(root):
 parent=parent_qualification(root)
 link=json.loads((Path(root)/'ADDITIONAL-PROOFS.json').read_text())
 for name,entry in link.items():
  paths=[Path(x)for x in entry['paths']]
  for p in paths:assert sha(p)==entry['sha256'][str(p)],str(p)
  result,launch,recovery,pins=[json.loads(p.read_text())for p in paths]
  for obj,key in ((launch,'returncode'),(recovery,'actual_returncode')):
   assert type(obj.get(key)) is int and obj[key]==0
   assert obj.get('child_absent') is True and obj.get('remaining_process_group')==[] and obj.get('lock_reacquired') is True
   assert not obj.get('error') and not obj.get('timeout')
  assert launch.get('finished') is not None and isinstance(recovery.get('post'),dict)
  assert launch['result_sha256']==recovery['result_sha256']==sha(paths[0]) and recovery['launch_sha256']==sha(paths[1])
  assert recovery['source_pin_file_sha256']==sha(paths[3]) and recovery['source_pins_verified']==len(pins)
  assert launch['pins']==pins
  for p,h in pins.items():assert sha(p)==h,p
  if name=='no_oracle_v1':
   from gdn_parent_state_no_oracle_v1.result_gate import audit as v1audit
   assert v1audit(result)['passed']
  else:
   assert name=='bool_not' and result.get('complete') is True and not result.get('error')
   assert len(result['records'])==16 and all(x['passed'] is True for x in result['records'])
   assert launch['result_gate']['raw_pass'] is True and recovery['result_gate']['raw_pass'] is True
 return dict(parent_proof=parent,no_oracle_v1=True,bool_not=True,native_invert_retained=False)
