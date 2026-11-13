"""CPU-default plan; parent-reviewed native-to-lean correctness only, no timing."""
import argparse,json,os,sys,signal
from pathlib import Path
from functools import partial
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from tileir_gdn_owned_replacement_v6 import adapter
from lean26_gdn_finite_runtime_v1.cleanup import final_cleanup
from lean26_gdn_finite_runtime_v1.gates import sha,qualification,frozen_reference,exact_tokens,exact_public,dispatch
from all_gemm_sixteen_sampler_composition_v2.cache_gate import preflight
from all_gemm_sixteen_sampler_composition_v2.precision import PrecisionPolicy
from packed_vllm_runtime_compare_v4.gates import public_logprobs
R=Path(__file__).resolve().parent;P=R.parent
PYTHON='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
IDS='/home/n/Documents/Codex/2026-10-31/task/tileir_poc/near40-model-search-20261109/g16-balanced-continuation-20261109/inputs/input-ids.json'
LOCK='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock'
def main():
 from tileir_gdn_owned_replacement_v6.preflight import source_preflight
 source_preflight(R)
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-gpu',action='store_true');ap.add_argument('--output',type=Path);a=ap.parse_args()
 ids=json.loads(Path(IDS).read_text());assert len(ids)==53 and sha(IDS)=='2843e98c58086b11e16679025ea499e97c360b2c4853ff0c211228ace09694dd'
 frozen_tokens,frozen_public,options=frozen_reference()
 result=dict(complete=False,gpu_executed=False,input_ids=ids,input_sha256=sha(IDS),engine_options=options,scope='current-eager native-to-lean correctness only',diagnostic_scope='GDN owned-copy qualified API replacement1152 with CPU byte oracle; native reference separate',timed_requests=0,expected_boundaries=5889,expected_logical_operations=6057,proof_scope='No new per-call X/Y raw/hash or W byte proof. End-to-end tokens/public values + census; historical combined intermediate raw proof stays separate.',baseline='Current eager vLLM mode0/graphsNONE B32P53G16 logprobs5 T/T; not historical optimized baseline',native_residuals=['retained public API and outer scope guards','remaining unclassified runner/output operations'],validation_scope='External proof diagnostics removed; public API checks and small input-domain checks remain')
 if not a.run_gpu:print(json.dumps(result,indent=2));return
 result['qualification_receipt']=qualification(P/'lean26_gdn_finite_runtime_v1') # First actual-run gate, before lock/import/model.
 assert os.environ.get('VLLM_COMPARISON_PARENT_APPROVED')=='1'and Path(sys.executable).absolute()==Path(PYTHON)
 assert a.output and a.output.is_absolute()and not a.output.exists()
 for p,h in json.loads((R/'PINS.json').read_text()).items():assert sha(p)==h,p
 held=os.fstat(int(os.environ['GEMM_COMPOSED_LOCK_FD']));target=Path(LOCK).stat();assert(held.st_dev,held.st_ino)==(target.st_dev,target.st_ino)
 result['qualification_receipt']=qualification(P/'lean26_gdn_finite_runtime_v1');result['cache_preseed']=preflight(R)
 a.output.mkdir()
 def save():(a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 def timeout(*_):raise TimeoutError('Bounded900s child; parent owns cleanup/recovery')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_ENABLE_V1_MULTIPROCESSING='0',DO_NOT_TRACK='1')
 result['runtime_environment']={k:v for k,v in os.environ.items()if k.startswith(('VLLM_','TORCHINDUCTOR_','TRITON_','FLASHINFER_','CUDA_'))or k in('XDG_CACHE_HOME','LD_LIBRARY_PATH','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')}
 engine=None;policy=None;installed=False
 try:
  import torch,vllm
  from vllm.engine.arg_utils import EngineArgs
  from vllm.engine.llm_engine import LLMEngine
  from vllm.sampling_params import SamplingParams,RequestOutputKind
  from vllm.usage.usage_lib import UsageContext
  assert vllm.__version__=='0.32.0'and torch.__version__=='2.13.0+cu130'
  from lean_finite_guard_runtime_v4.edge_probe import run as edge_probe
  result.update(gpu_executed=True,phase='finite_edge_probe');save()
  result['finite_edge_probe']=edge_probe();save()
  policy=PrecisionPolicy(torch);result['precision_policy']=policy.record;result.update(gpu_executed=True,phase='engine_init',torch_version=torch.__version__,vllm_version=vllm.__version__);save()
  engine=LLMEngine.from_engine_args(EngineArgs(**options),usage_context=UsageContext.LLM_CLASS,enable_multiprocessing=False)
  torch.cuda.synchronize();cfg=engine.vllm_config
  assert type(engine.engine_core).__name__=='InprocClient'and cfg.model_config.enforce_eager
  assert int(cfg.compilation_config.mode)==0 and cfg.compilation_config.cudagraph_mode.name=='NONE'and cfg.scheduler_config.async_scheduling is True
  result['resolved_config']={k:str(getattr(cfg,k+'_config'))for k in('model','scheduler','cache','compilation','parallel')};result['resolved_async_scheduling']=cfg.scheduler_config.async_scheduling
  params=SamplingParams(n=1,temperature=0,min_tokens=16,max_tokens=16,ignore_eos=False,detokenize=False,logprobs=5,prompt_logprobs=None,seed=20261111,output_kind=RequestOutputKind.CUMULATIVE)
  def request(mode):
   assert mode in('native','lean')and not engine.has_unfinished_requests()
   _,precision=policy.select('native')
   if mode=='lean':engine.apply_model(partial(adapter.start,mode='lean'))
   keys=[mode+':'+str(i)for i in range(32)];outputs={};torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
   for key in keys:engine.add_request(key,{'prompt_token_ids':list(ids)},params)
   while engine.has_unfinished_requests():
    for o in engine.step():
     if o.finished:
      assert o.request_id in keys and o.request_id not in outputs and len(o.outputs)==1;outputs[o.request_id]=o
   torch.cuda.synchronize();assert set(outputs)==set(keys)
   tokens=[list(outputs[k].outputs[0].token_ids)for k in keys];public=public_logprobs(outputs,keys)
   item=dict(mode=mode,output_token_ids=tokens,public_logprobs=public,selected_dispatch=None,allow_bf16_reduced_precision_reduction=precision,allow_bf16_reduced_precision_reduction_split_k=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
   result[mode+'_request']=item;save() # retain exact failing outputs before comparisons
   item['frozen_tokens_gate']=exact_tokens(tokens,frozen_tokens);item['frozen_public_gate']=exact_public(public,frozen_public)
   if mode=='lean':
    d=engine.apply_model(adapter.finish);assert len(d)==1;item['selected_dispatch']=d[0];item['dispatch_gate']=dispatch(d[0])
   save();return item
  result['phase']='native_reference';save();native=request('native')
  assert not engine.has_unfinished_requests();torch.cuda.synchronize()
  result['phase']='install_lean';save();installed=True
  result['adapter_install']=engine.apply_model(adapter.install);assert len(result['adapter_install'])==1
  result['phase']='lean_correctness';save();lean=request('lean')
  result['same_run_token_gate']=exact_tokens(native['output_token_ids'],lean['output_token_ids'])
  result['same_run_public_gate']=exact_public(native['public_logprobs'],lean['public_logprobs'])
  result.update(candidate_checks_passed=True,phase='candidate_checks_complete');save()
 except BaseException as e:
  result.update(error=repr(e),complete=False,failure_type=type(e).__name__,automatic_retry=False,runtime_settings_changed=False);save();raise
 finally:
  final_cleanup(result,save,policy,engine,adapter,installed,lambda:__import__('torch').cuda.synchronize(),lambda:signal.alarm(0),propagating=sys.exc_info()[0]is not None)
if __name__=='__main__':main()
