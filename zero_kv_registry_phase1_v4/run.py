import argparse,json,os,sys,signal,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from zero_kv_registry_phase1_v4.wrapper import Hooks
from zero_kv_registry_phase1_v4.registry import Hold
from zero_kv_registry_phase1_v4.gates import qualification
from zero_kv_registry_phase1_v4.runtime_contract import runtime_guard,record_endpoints
PYTHON='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
LOCK='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock'
IDS='/home/n/Documents/Codex/2026-10-31/task/tileir_poc/near40-model-search-20261109/g16-balanced-continuation-20261109/inputs/input-ids.json'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-gpu',action='store_true');ap.add_argument('--output',type=Path);a=ap.parse_args()
 if not a.run_gpu:print(json.dumps(dict(executed=False,scope='ZeroKV192B registry geometry only; union>256MiB HOLD before nativezero',framework_imported=False)));return 0
 from lean_public_runtime_qualification_v2.gates import frozen_reference,exact_tokens,exact_public
 from packed_vllm_runtime_compare_v4.gates import public_logprobs
 from all_gemm_sixteen_sampler_composition_v2.cache_gate import preflight
 from all_gemm_sixteen_sampler_composition_v2.precision import PrecisionPolicy
 ft,fp,options=frozen_reference()
 qualification(R);cache=preflight(R)
 assert os.environ.get('VLLM_COMPARISON_PARENT_APPROVED')=='1' and Path(sys.executable).absolute()==Path(PYTHON)
 fd=int(os.environ['GEMM_COMPOSED_LOCK_FD']);s=os.fstat(fd);t=Path(LOCK).stat();assert(s.st_dev,s.st_ino)==(t.st_dev,t.st_ino)
 for p,h in json.loads((R/'PINS.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 assert a.output and a.output.is_absolute() and not a.output.exists();a.output.mkdir()
 result=dict(complete=False,scope='phase1 owner/table geometry only; native zero after guards; no TileIR zero',cache_preseed=cache,engine_options=options,cleanup_errors=[]);hooks=Hooks();engine=None;policy=None;code=1
 def save():result['diagnostic']=hooks.receipt();(a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(Hold('timeout')));signal.alarm(900)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_ENABLE_V1_MULTIPROCESSING='0',DO_NOT_TRACK='1')
 try:
  import torch,vllm
  import vllm.v1.worker.gpu.model_runner as mr
  import vllm.v1.worker.utils as u
  from vllm.engine.arg_utils import EngineArgs
  from vllm.engine.llm_engine import LLMEngine
  from vllm.sampling_params import SamplingParams,RequestOutputKind
  from vllm.usage.usage_lib import UsageContext
  assert vllm.__version__=='0.32.0' and torch.__version__=='2.13.0+cu130';policy=PrecisionPolicy(torch);hooks.install(mr,u);save()
  engine=LLMEngine.from_engine_args(EngineArgs(**options),usage_context=UsageContext.LLM_CLASS,enable_multiprocessing=False);torch.cuda.synchronize();assert type(engine.engine_core).__name__=='InprocClient'
  result['resolved_runtime']=runtime_guard(engine,policy);result['precision_policy']=policy.record
  hooks.begin_request();save();ids=json.loads(Path(IDS).read_text());assert len(ids)==53
  params=SamplingParams(n=1,temperature=0,min_tokens=16,max_tokens=16,ignore_eos=False,detokenize=False,logprobs=5,prompt_logprobs=None,seed=20261111,output_kind=RequestOutputKind.CUMULATIVE)
  keys=['zero_registry:'+str(i)for i in range(32)];outputs={}
  for key in keys:engine.add_request(key,{'prompt_token_ids':ids},params)
  while engine.has_unfinished_requests():
   for o in engine.step():
    if o.finished:outputs[o.request_id]=o
  torch.cuda.synchronize();tokens=[list(outputs[k].outputs[0].token_ids)for k in keys];public=public_logprobs(outputs,keys)
  record_endpoints(result,tokens,public,save,exact_tokens,exact_public,ft,fp)
  assert hooks.diag.native_zero_calls==hooks.diag.native_zero_returned==1 and hooks.diag.receipt['within_budget'];result['checks_passed']=True;code=0
 except BaseException as e:
  from zero_kv_registry_phase1_v4.diagnostic import safe_code
  result['error_code']=safe_code(e,'engine_or_boundary_failure');result['hold']=True;code=2
 finally:
  def cleanup(label,fn):
   try:fn();result[label]=True;return True
   except BaseException:result['cleanup_errors'].append(label);return False
  cleanup('hooks_restored',hooks.close)
  synced=cleanup('synchronized',lambda:__import__('torch').cuda.synchronize())
  if synced and hooks.restored:cleanup('owners_released',lambda:hooks.release(after_synchronize=True))
  if policy is not None:cleanup('precision_restored',policy.restore)
  if engine is not None:cleanup('engine_shutdown',engine.engine_core.shutdown)
  signal.alarm(0);result['complete']=code==0 and not result['cleanup_errors'];save()
 return code if not result['cleanup_errors'] else 1
if __name__=='__main__':raise SystemExit(main())
