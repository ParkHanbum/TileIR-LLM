"""CPU-default preparation. GPU execution requires reviewed parent lifecycle/admission.
Correctness-only eager public attention failure boundary and late-prefix capture; no performance claim.
"""
import argparse,hashlib,json,os,sys,time,signal
from pathlib import Path
from cleanup import final_cleanup
P=Path(__file__).resolve().parent.parent
MODEL='/home/n/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
IDS='/home/n/Documents/Codex/2026-10-31/task/tileir_poc/near40-model-search-20261109/g16-balanced-continuation-20261109/inputs/input-ids.json'
PYTHON='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
LOCK='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 from preflight import verify
 verify(Path(__file__).resolve().parent,cache=False)
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-gpu',action='store_true');ap.add_argument('--output',type=Path);ap.add_argument('--expected-tokens',type=Path);ap.add_argument('--sync-step-boundary',action='store_true');ap.add_argument('--eager-diagnostic',action='store_true');a=ap.parse_args()
 ids=json.loads(Path(IDS).read_text());assert len(ids)==53 and sha(IDS)=='2843e98c58086b11e16679025ea499e97c360b2c4853ff0c211228ace09694dd'
 options=dict(model=MODEL,tokenizer=MODEL,dtype='bfloat16',tensor_parallel_size=1,max_model_len=128,max_num_seqs=32,max_num_batched_tokens=2048,gpu_memory_utilization=0.85,enable_prefix_caching=False,enable_chunked_prefill=False,enforce_eager=True,trust_remote_code=False,limit_mm_per_prompt={'image':0,'video':0},disable_log_stats=False)
 assert not a.sync_step_boundary,'Keep original async policy'
 options['compilation_config']={'mode':0,'cudagraph_mode':'NONE'}
 result=dict(complete=False,gpu_executed=False,model=MODEL,input_ids=ids,input_sha256=sha(IDS),engine_options=options,scope='Eager compilation NONE / graphs NONE public attention failure boundary and late-prefix observation; not optimized vLLM',output_contract='Token-only, detokenizeFalse, no logits/logprobs retention; all fixed512 token IDs must equal EXPECTED-TOKENS.json and within-runtime repetitions must match',decode_caveat='Engine first/last-token timestamps are distinct from TileIR first-incremental-forward synchronized boundary',warmup_requests=1,timed_requests=0,correctness_only=True,capture_requests=1)
 if not a.run_gpu:print(json.dumps(result,indent=2));return
 verify(Path(__file__).resolve().parent,cache=True)
 assert os.environ.get('VLLM_COMPARISON_PARENT_APPROVED')=='1'
 assert Path(sys.executable).absolute()==Path(PYTHON)
 assert a.output and a.output.is_absolute() and not a.output.exists()
 pins_path=Path(__file__).with_name('PINS.json');assert pins_path.exists(),'Pending independent review/final lifecycle launcher'
 for p,h in json.loads(pins_path.read_text()).items():assert sha(p)==h,p
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);held=os.fstat(fd);target=Path(LOCK).stat();assert (held.st_dev,held.st_ino)==(target.st_dev,target.st_ino)
 expected=json.loads(Path(__file__).with_name('EXPECTED-TOKENS.json').read_text())
 if expected is not None:assert len(expected)==32 and all(len(x)==16 for x in expected)
 a.output.mkdir();result['expected_tokens_sha256']=sha(Path(__file__).with_name('EXPECTED-TOKENS.json'))
 def save():(a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 def timeout(*_):raise TimeoutError('Bounded900s child; parent owns kill/recovery')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_ENABLE_V1_MULTIPROCESSING='0',DO_NOT_TRACK='1')
 result['runtime_environment']={k:v for k,v in os.environ.items() if k.startswith(('VLLM_','TORCHINDUCTOR_','TRITON_','FLASHINFER_','CUDA_')) or k in ('XDG_CACHE_HOME','LD_LIBRARY_PATH','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')}
 engine=None;close_attempted=False;observer_live=False;check_flags=None
 import observer
 try:
  import torch,vllm
  from vllm.engine.arg_utils import EngineArgs
  from vllm.engine.llm_engine import LLMEngine
  from vllm.sampling_params import SamplingParams,RequestOutputKind
  from vllm.usage.usage_lib import UsageContext
  assert vllm.__version__=='0.32.0' and torch.__version__=='2.13.0+cu130'
  result.update(gpu_executed=True,phase='engine_init',torch_version=torch.__version__,vllm_version=vllm.__version__);save()
  init_start=time.perf_counter_ns()
  engine=LLMEngine.from_engine_args(EngineArgs(**options),usage_context=UsageContext.LLM_CLASS,enable_multiprocessing=False)
  torch.cuda.synchronize();result['initialization_including_compile_graph_ms']=(time.perf_counter_ns()-init_start)/1e6
  cfg=engine.vllm_config
  result['initial_math_flags']=[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
  assert result['initial_math_flags']==[True,True]
  def check_flags():
   assert result['initial_math_flags']==[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
   result['math_flags_unchanged']=True
  result['resolved_async_scheduling']=cfg.scheduler_config.async_scheduling
  result['resolved_config']=dict(model=str(cfg.model_config),scheduler=str(cfg.scheduler_config),cache=str(cfg.cache_config),compilation=str(cfg.compilation_config),parallel=str(cfg.parallel_config),core_client_class=type(engine.engine_core).__name__)
  assert type(engine.engine_core).__name__=='InprocClient','Device synchronization must cover owning process'
  if a.sync_step_boundary:assert cfg.scheduler_config.async_scheduling is False
  assert cfg.model_config.enforce_eager and cfg.compilation_config.mode.name=='NONE'
  assert cfg.compilation_config.cudagraph_mode.name=='NONE'
  result['diagnostic_policy']='enforce_eager: compilation AND graphs OFF; same public Linear implementations, not optimized-runtime capture'
  assert cfg.scheduler_config.async_scheduling is True
  params=SamplingParams(n=1,temperature=0,min_tokens=16,max_tokens=16,ignore_eos=False,detokenize=False,logprobs=None,prompt_logprobs=None,seed=20261111,output_kind=RequestOutputKind.CUMULATIVE)
  repeat_reference=None
  def request(index):
   nonlocal repeat_reference
   assert not engine.has_unfinished_requests()
   keys=[f'{index}:{i}' for i in range(32)];latest={};steps=[];first_step_end=None;first_public_output_ns=None;final_metrics={}
   result['last_request_steps']=steps
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter_ns()
   for key in keys:engine.add_request(key,{'prompt_token_ids':list(ids)},params)
   if not a.sync_step_boundary:
    # Public engine stepping is the API; no per-step timing, assertions or metrics.
    final_outputs={}
    while engine.has_unfinished_requests():
     for o in engine.step():
      if o.finished:final_outputs[o.request_id]=o
    output=[list(final_outputs[k].outputs[0].token_ids) for k in keys]
    torch.cuda.synchronize();end=time.perf_counter_ns()
    # Diagnostics and validation are outside the primary wall.
    for k,o in final_outputs.items():
     assert k in keys and len(o.outputs)==1
     latest[k]=list(o.outputs[0].token_ids)
     if o.metrics is not None:
      final_metrics[k]={name:getattr(o.metrics,name,None) for name in ['first_token_ts','last_token_ts','num_generation_tokens','num_preemptions']}
   else:
    while engine.has_unfinished_requests():
     outputs=engine.step();row={}
     if outputs and first_public_output_ns is None:first_public_output_ns=time.perf_counter_ns()
     for o in outputs:
      assert o.request_id in keys and len(o.outputs)==1
      tokens=list(o.outputs[0].token_ids);latest[o.request_id]=tokens;row[o.request_id]=len(tokens)
     steps.append(row)
     if len(steps)==1:
      assert set(row)==set(keys) and set(row.values())=={1},'First public step did not yield32x1; no decode estimate'
      torch.cuda.synchronize();first_step_end=time.perf_counter_ns()
    torch.cuda.synchronize();end=time.perf_counter_ns()
   output=[latest[k] for k in keys];result['last_output_token_ids']=output
   if repeat_reference is None:repeat_reference=output
   assert output==repeat_reference,'Within-runtime repeated tokens diverged; outputs preserved' 
   assert all(len(x)==16 for x in output)
   item=dict(cross_runtime_tokens_equal=None if expected is None else output==expected,first_public_output_to_synchronized_completion_ms=None if first_public_output_ns is None else (end-first_public_output_ns)/1e6,first_output_interval_scope='Frontend observation only; asyncqueued decode may already be running, not puredecode',whole_ms=(end-start)/1e6,wall_scope='32 public requests submitted through final CPU token IDs ready plus device synchronization; no per-step diagnostic in primary',generated_tokens=sum(map(len,output)),steps=steps,output_token_ids=output,engine_request_metrics=final_metrics,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
   if a.sync_step_boundary:
    assert len(steps)==16 and all(set(row)==set(keys) and set(row.values())=={j+1} for j,row in enumerate(steps))
    item.update(synchronized_public_step_decode_ms=(end-first_step_end)/1e6,decode_tokens=480,decode_boundary='After first public engine.step32x1 output plus device sync, through final engine output andsync; includes later scheduling/output processing, not identical to TileIR preforward boundary')
   return item
  result['phase']='warmup';save()
  result['warmup_records']=[]
  for i in range(1):result['warmup_records'].append(request('warm'+str(i)));save()
  result['first_request_is_cold']=True
  result['phase']='correctness_capture';result['samples']=[];save()
  from functools import partial
  engine.apply_model(partial(observer.install,out=str(a.output/'bank')));observer_live=True
  try:
   result['samples'].append(request('capture0'));save()
  finally:
   result['capture_manifest']=engine.apply_model(observer.finish);observer_live=False;save()
  assert len(result['capture_manifest'])==1 and result['capture_manifest'][0]['complete']
  assert result['capture_manifest'][0]['observers_removed']
  assert all(x['cross_runtime_tokens_equal'] is True for x in result['warmup_records']+result['samples'])
  assert result['initial_math_flags']==[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
  result['math_flags_unchanged']=True
  torch.cuda.synchronize();close_attempted=True;engine.engine_core.shutdown();result['engine_shutdown']=True;engine=None
  result.update(complete=True,phase='complete');save()
 except BaseException as e:
  result.update(error=repr(e),complete=False);save();raise
 finally:
  final_cleanup(result,save,check_flags,engine,observer,observer_live,close_attempted,
                lambda: __import__('torch').cuda.synchronize(),lambda: signal.alarm(0),
                propagating=sys.exc_info()[0] is not None)
if __name__=='__main__':main()
