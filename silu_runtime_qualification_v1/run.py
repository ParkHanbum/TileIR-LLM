"""Small wrapper: reuse the frozen comparison, preserve attempt/error evidence."""
import argparse,fcntl,json,os,signal,sys,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
MANIFEST=P/'packed_runtime_contract_capture_v1/packedCapture01/bank/MANIFEST.json'
MANIFEST_SHA='719a7fb82ab3f9be0a9ee8383226a47811bf5ba8686542347a0c3b6e35bd6a08'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-gpu',action='store_true');ap.add_argument('--output',required=True);a=ap.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(MANIFEST)==MANIFEST_SHA
 if not a.run_gpu:print(json.dumps(dict(executed=False,manifest_sha=MANIFEST_SHA)));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);lock='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock'
 assert os.fstat(fd).st_ino==os.stat(lock).st_ino;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=Path(a.output);out.mkdir(exist_ok=False)
 for key,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (out/name).mkdir();os.environ[key]=str(out/name)
 r=dict(complete=False,gpu_executed=False,manifest_sha=MANIFEST_SHA,records=[],timing=False,stage='wrapper_started',started=time.time())
 def save():(out/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.alarm(175)
 try:
  from tileir_vllm_silu_mul_v2 import compare_captured
  r.update(stage='comparison_invocation_intent',gpu_executed=None,execution_intent=True);save()
  # This is intent, not proof a target kernel started. Original comparison is
  # unchanged; failure stderr identifies the last reached Python operation.
  sys.argv=[str(compare_captured.__file__),'--execute-gpu','--output',str(out/'RAW.json')]
  compare_captured.main()
  raw=json.loads((out/'RAW.json').read_text());r.update(raw);r.update(complete=True,stage='comparison_returned',raw_sha=sha(out/'RAW.json'))
  verify_pins(R/'CPU-PINS.json');save()
  from silu_runtime_qualification_v1.launch import result_gate
  gate=result_gate(r);r['gate']=gate;save()
  return 0 if gate['raw_pass'] else (2 if gate['raw_difference_hold'] else 1)
 except BaseException as e:
  r.update(error=repr(e),traceback=traceback.format_exc(),stage='failed_after_intent',outcome='partial/unknown target progress; no automatic retry')
  save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
