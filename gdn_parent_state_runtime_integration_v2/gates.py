from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
from pathlib import Path
import json,hashlib
RESULT_SHA='ce88fc3f48a237c739d121606761bc1231093e3daa20491c97741cd387c5931d'
LAUNCH_SHA='56afcc171a1597243e9e95cf3e13e75e950cf3d6a0c277993f4d90a405a28122'
RECOVERY_SHA='0d83e2ed7491ccad00d0227f8dcfc5da9a98948e936896f3b25c03379bf967c9'
PINS_SHA='6b41c2b7d9195ada5275a6c65ffbd1eea6d1bd03718f1d088f4958072eb6ec52'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_receipts(result,launch,recovery):
 assert result.get('complete') is True and not result.get('error') and not result.get('hold')
 records=result.get('records');assert isinstance(records,list) and len(records)==12 and all(x.get('passed') is True and x.get('mask_true_bits_preserved_false_positive_zero') is True for x in records)
 assert type(launch.get('returncode')) is int and launch['returncode']==0
 assert type(recovery.get('actual_returncode')) is int and recovery['actual_returncode']==0
 for r in (launch,recovery):
  assert r.get('child_absent') is True and r.get('remaining_process_group')==[] and r.get('lock_reacquired') is True
  assert r.get('result_sha256')==RESULT_SHA and r.get('result_gate',{}).get('raw_pass') is True
  assert r['result_gate'].get('raw_difference_hold') is False
  assert not r.get('error') and not r.get('timeout')
 assert launch.get('finished') is not None
 assert recovery.get('launch_sha256')==LAUNCH_SHA and recovery.get('source_pin_file_sha256')==PINS_SHA
 assert recovery.get('source_pins_verified')==2741 and isinstance(recovery.get('post'),dict)
 return True

def qualification(root):
 base=Path(root).parent/'gdn_parent_state_synthetic_v2'
 paths=[base/'gdnParentStateSynthetic02/RESULT.json',base/'gdnParentStateSynthetic02-launch.json',base/'gdnParentStateSynthetic02-recovery.json',base/'CPU-PINS.json']
 for p,h in zip(paths,(RESULT_SHA,LAUNCH_SHA,RECOVERY_SHA,PINS_SHA)):assert sha(p)==h,str(p)
 result,launch,recovery,pins=[json.loads(p.read_text()) for p in paths]
 validate_receipts(result,launch,recovery)
 assert len(pins)==2741 and launch['pins']==pins
 for p,h in pins.items():assert sha(p)==h,('Synthetic source changed',p)
 return dict(synthetic_pass=True,actual_exit0=True,independent_recovery=True,source_pins_verified=len(pins),actual_model_parent_pending=True,receipts={str(p):sha(p) for p in paths})
