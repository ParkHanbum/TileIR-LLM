"""CPU-default current-eager18-controller balanced wall and separate minimal profiler."""
import argparse,json,os,sys,signal,gc
from pathlib import Path
from functools import partial
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from lean_public_composition_v1 import adapter
from lean_public_composition_v1.support import Base
from lean_public_runtime_qualification_v1.cleanup import final_cleanup
from lean_public_runtime_qualification_v1.gates import sha,frozen_reference
from all_gemm_sixteen_sampler_composition_v2.cache_gate import preflight
from all_gemm_sixteen_sampler_composition_v2.precision import PrecisionPolicy
from lean_public_balanced_measure_v2.design import orders,profile_order
from lean_public_balanced_measure_v2.scopes import public_api_scopes
from lean_public_balanced_measure_v2.gates import qualification
from lean_public_balanced_measure_v2.request import request,absent
from lean_public_balanced_measure_v2.profile_contract import inspect_trace
R=Path(__file__).resolve().parent;P=R.parent
PYTHON='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
IDS='/home/n/Documents/Codex/2026-10-31/task/tileir_poc/near40-model-search-20261109/g16-balanced-continuation-20261109/inputs/input-ids.json'
LOCK='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock'
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-gpu',action='store_true');ap.add_argument('--output',type=Path);ap.add_argument('--process-index',type=int,choices=(0,1),default=0);a=ap.parse_args();C=R/('process'+str(a.process_index))
 ids=json.loads(Path(IDS).read_text());assert len(ids)==53 and sha(IDS)=='2843e98c58086b11e16679025ea499e97c360b2c4853ff0c211228ace09694dd'
 ft,fp,options=frozen_reference()
 result=dict(complete=False,gpu_executed=False,process_index=a.process_index,baseline='current eager vLLM mode0/graphsNONE',engine_options=options,input_ids=ids,input_sha256=sha(IDS),orders=orders(a.process_index),profile_order=profile_order(a.process_index),qualification_requests=[],pairs=[],profile_requests=[],wall_scope='Before32add_request calls through final CPU engine outputs/token IDs and device completion; installation, proof formatting/gates and restoration excluded',scope='Exploratory warm-cache public-API-cost comparison of fixed18-controller lean; no19-controller additions, no100% coverage',state_reset_scope='Fresh request IDs, no unfinished requests and prefix caching disabled; rely on native vLLM request/KV initialization, no manual state byte reset',remaining_native=['attention core','GDN prefill','cat/output assembly','remaining runner work'],profile_scope='Separate CPU+CUDA activity trace; record shapes/memory/stack/flops disabled; profile-only public API ranges. Only request_execution-correlated launches belong to the model denominator; install/proof/restore are excluded. Public API ranges still do not themselves distinguish validator kernels from numerical TileIR kernels.')
 if not a.run_gpu:print(json.dumps(result,indent=2));return
 assert os.environ.get('VLLM_COMPARISON_PARENT_APPROVED')=='1'and Path(sys.executable).absolute()==Path(PYTHON)
 assert a.output and a.output.is_absolute()and a.output.parent==C and not a.output.exists()
 for p,h in json.loads((R/'PINS.json').read_text()).items():assert sha(p)==h,p
 held=os.fstat(int(os.environ['GEMM_COMPOSED_LOCK_FD']));target=Path(LOCK).stat();assert(held.st_dev,held.st_ino)==(target.st_dev,target.st_ino)
 result['qualification_receipt']=qualification(R);result['cache_preseed']=preflight(C);a.output.mkdir()
 def save():(a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 def timeout(*_):raise TimeoutError('Bounded900s child; no automatic retry')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_ENABLE_V1_MULTIPROCESSING='0',DO_NOT_TRACK='1')
 result['runtime_environment']={k:v for k,v in os.environ.items()if k.startswith(('VLLM_','TORCHINDUCTOR_','TRITON_','FLASHINFER_','CUDA_'))or k in('XDG_CACHE_HOME','LD_LIBRARY_PATH','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')}
 engine=None;policy=None;lifecycle={'installed':False};base_call=Base.call
 try:
  import torch,vllm
  from vllm.engine.arg_utils import EngineArgs
  from vllm.engine.llm_engine import LLMEngine
  from vllm.sampling_params import SamplingParams,RequestOutputKind
  from vllm.usage.usage_lib import UsageContext
  assert vllm.__version__=='0.32.0'and torch.__version__=='2.13.0+cu130'
  policy=PrecisionPolicy(torch);result['precision_policy']=policy.record;result.update(gpu_executed=True,phase='engine_init',torch_version=torch.__version__,vllm_version=vllm.__version__);save()
  engine=LLMEngine.from_engine_args(EngineArgs(**options),usage_context=UsageContext.LLM_CLASS,enable_multiprocessing=False);torch.cuda.synchronize();cfg=engine.vllm_config
  assert type(engine.engine_core).__name__=='InprocClient'and cfg.model_config.enforce_eager and int(cfg.compilation_config.mode)==0 and cfg.compilation_config.cudagraph_mode.name=='NONE'and cfg.scheduler_config.async_scheduling is True
  result['resolved_config']={k:str(getattr(cfg,k+'_config'))for k in('model','scheduler','cache','compilation','parallel')}
  params=SamplingParams(n=1,temperature=0,min_tokens=16,max_tokens=16,ignore_eos=False,detokenize=False,logprobs=5,prompt_logprobs=None,seed=20261111,output_kind=RequestOutputKind.CUMULATIVE)
  def rec(x):result['last_request']=x;save()
  def run(mode,label,measured,profiled=False):return request(engine,torch,policy,adapter,lifecycle,ids,params,label,mode,measured,ft,fp,rec,profiled=profiled)
  result['phase']='untimed_same_process_qualification_and_warmup';save()
  for mode in('native','lean'):result['qualification_requests'].append(run(mode,'qual_'+mode,False));save()
  result['phase']='balanced_wall';save()
  for i,order in enumerate(orders(a.process_index)):
   row=dict(pair_index=i,order=order,requests=[]);result['pairs'].append(row)
   for mode in order:
    assert Base.call is base_call,'Profile scope must not leak into wall samples'
    row['requests'].append(run(mode,'wall_'+str(i)+'_'+mode,True));save()
  result['phase']='separate_profiles';save()
  for mode in profile_order(a.process_index):
   with public_api_scopes(torch,mode=='lean'):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False,with_flops=False)as prof:
     item=run(mode,'profile_'+mode,False,profiled=True)
   assert Base.call is base_call
   trace=a.output/(mode+'.trace.json');prof.export_chrome_trace(str(trace));torch.cuda.synchronize();del prof;gc.collect()
   size=trace.stat().st_size;assert 0<size<=128*1024*1024,'Trace over128MiB cap; preserve and HOLD'
   item.update(profiled=True,trace_path=str(trace),trace_bytes=size,trace_sha256=sha(trace));result['profile_requests'].append(item);save()
   item['profile_partition']=inspect_trace(trace,mode);save();assert item['profile_partition']['model_denominator_ready'],'Request launch-correlation denominator HOLD; session total must not substitute'
  result['final_adapter_absent']=engine.apply_model(partial(absent,adapter=adapter));assert result['final_adapter_absent']==[True]and not lifecycle['installed']and Base.call is base_call
  result.update(candidate_checks_passed=True,profile_scope_restored=True,phase='measurements_complete');save()
 except BaseException as e:result.update(error=repr(e),complete=False,failure_type=type(e).__name__,automatic_retry=False,runtime_settings_changed=False);save();raise
 finally:
  final_cleanup(result,save,policy,engine,adapter,lifecycle['installed'],lambda:__import__('torch').cuda.synchronize(),lambda:signal.alarm(0),propagating=sys.exc_info()[0]is not None)
if __name__=='__main__':main()
