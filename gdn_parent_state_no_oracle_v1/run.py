"""CPU-default preparation. GPU execution requires reviewed parent lifecycle/admission.
Eager correctness only; no optimized fusion or performance claim.
"""
import argparse,hashlib,json,os,sys,time,signal
from pathlib import Path
from functools import partial
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from public_vllm_composed_integration_v5.cleanup import final_cleanup
from gdn_parent_state_no_oracle_v1.precision import PrecisionPolicy,ORDERS
from gdn_parent_state_no_oracle_v1 import adapter
from gdn_parent_state_no_oracle_v1.result_gate import native_finish_gate
from gdn_parent_state_no_oracle_v1.cache_gate import preflight
from gdn_parent_state_no_oracle_v1.gates import public_logprobs,compare_public,qualification
P=Path(__file__).resolve().parent.parent
MODEL='/home/n/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
IDS='/home/n/Documents/Codex/2026-10-31/task/tileir_poc/near40-model-search-20261109/g16-balanced-continuation-20261109/inputs/input-ids.json'
PYTHON='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
LOCK='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-gpu',action='store_true');ap.add_argument('--output',type=Path);ap.add_argument('--expected-tokens',type=Path);ap.add_argument('--sync-step-boundary',action='store_true');ap.add_argument('--eager-diagnostic',action='store_true');a=ap.parse_args()
 ids=json.loads(Path(IDS).read_text());assert len(ids)==53 and sha(IDS)=='2843e98c58086b11e16679025ea499e97c360b2c4853ff0c211228ace09694dd'
 options=dict(model=MODEL,tokenizer=MODEL,dtype='bfloat16',tensor_parallel_size=1,max_model_len=128,max_num_seqs=32,max_num_batched_tokens=2048,gpu_memory_utilization=0.85,enable_prefix_caching=False,enable_chunked_prefill=False,enforce_eager=True,trust_remote_code=False,limit_mm_per_prompt={'image':0,'video':0},disable_log_stats=False)
 options['compilation_config']={'mode':0,'cudagraph_mode':'NONE'}
 assert not a.sync_step_boundary,'Both arms retain the same default async scheduler'
 result=dict(complete=False,gpu_executed=False,model=MODEL,input_ids=ids,input_sha256=sha(IDS),engine_options=options,scope='Native baseline24 then TileIR parent24 with no target oracle; native invert/core retained',output_contract='Two correctness requests, public top5/chosen logprobs retained; no timing/performance claim',decode_caveat='Engine first/last-token timestamps are distinct from TileIR first-incremental-forward synchronized boundary',warmup_requests=0,timed_requests=0)
 if not a.run_gpu:print(json.dumps(result,indent=2));return
 assert os.environ.get('VLLM_COMPARISON_PARENT_APPROVED')=='1'
 assert Path(sys.executable).absolute()==Path(PYTHON)
 assert a.output and a.output.is_absolute() and not a.output.exists()
 pins_path=Path(__file__).with_name('PINS.json');assert pins_path.exists(),'Pending independent review/final lifecycle launcher'
 for p,h in json.loads(pins_path.read_text()).items():assert sha(p)==h,p
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);held=os.fstat(fd);target=Path(LOCK).stat();assert (held.st_dev,held.st_ino)==(target.st_dev,target.st_ino)
 expected_path=Path(__file__).with_name('EXPECTED-TOKENS.json');expected=json.loads(expected_path.read_text())
 if expected is not None:assert len(expected)==32 and all(len(x)==16 for x in expected)
 a.output.mkdir();result['expected_tokens_sha256']=sha(expected_path)
 def save():(a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 def timeout(*_):raise TimeoutError('Bounded900s child; parent owns kill/recovery')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_ENABLE_V1_MULTIPROCESSING='0',DO_NOT_TRACK='1')
 result['runtime_environment']={k:v for k,v in os.environ.items() if k.startswith(('VLLM_','TORCHINDUCTOR_','TRITON_','FLASHINFER_','CUDA_')) or k in ('XDG_CACHE_HOME','LD_LIBRARY_PATH','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')}
 result['qualification_receipt']=qualification(Path(__file__).resolve().parent);result['cache_preseed']=preflight(Path(__file__).resolve().parent);save()
 engine=None;close_attempted=False;adapter_installed=False;policy=None
 try:
  import torch,vllm
  from vllm.engine.arg_utils import EngineArgs
  from vllm.engine.llm_engine import LLMEngine
  from vllm.sampling_params import SamplingParams,RequestOutputKind
  from vllm.usage.usage_lib import UsageContext
  assert vllm.__version__=='0.32.0' and torch.__version__=='2.13.0+cu130'
  policy=PrecisionPolicy(torch);result['precision_policy']=policy.record
  result.update(gpu_executed=True,phase='engine_init',torch_version=torch.__version__,vllm_version=vllm.__version__);save()
  init_start=time.perf_counter_ns()
  engine=LLMEngine.from_engine_args(EngineArgs(**options),usage_context=UsageContext.LLM_CLASS,enable_multiprocessing=False)
  torch.cuda.synchronize();result['initialization_including_compile_graph_ms']=(time.perf_counter_ns()-init_start)/1e6
  cfg=engine.vllm_config
  result['resolved_async_scheduling']=cfg.scheduler_config.async_scheduling
  result['resolved_config']=dict(model=str(cfg.model_config),scheduler=str(cfg.scheduler_config),cache=str(cfg.cache_config),compilation=str(cfg.compilation_config),parallel=str(cfg.parallel_config),core_client_class=type(engine.engine_core).__name__)
  assert type(engine.engine_core).__name__=='InprocClient','Device synchronization must cover owning process'
  if a.sync_step_boundary:assert cfg.scheduler_config.async_scheduling is False
  assert cfg.model_config.enforce_eager
  assert int(cfg.compilation_config.mode)==0 and cfg.compilation_config.cudagraph_mode.name=='NONE'
  result['adapter_install']=engine.apply_model(partial(adapter.install,output=str(a.output/'shadow')));adapter_installed=True
  params=SamplingParams(n=1,temperature=0,min_tokens=16,max_tokens=16,ignore_eos=False,detokenize=False,logprobs=None,prompt_logprobs=None,seed=20261111,output_kind=RequestOutputKind.CUMULATIVE)
  gate_params=SamplingParams(n=1,temperature=0,min_tokens=16,max_tokens=16,ignore_eos=False,detokenize=False,logprobs=5,prompt_logprobs=None,seed=20261111,output_kind=RequestOutputKind.CUMULATIVE)
  repeat_reference=None
  def request(index,mode,diagnostic=False):
   nonlocal repeat_reference
   assert not engine.has_unfinished_requests()
   backend,precision_setting=policy.select('native')
   engine.apply_model(partial(adapter.start,mode=mode))
   keys=[f'{index}:{i}' for i in range(32)];latest={};steps=[];first_step_end=None;first_public_output_ns=None;final_metrics={}
   result['last_request_steps']=steps
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter_ns()
   for key in keys:engine.add_request(key,{'prompt_token_ids':list(ids)},gate_params if diagnostic else params)
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
   counts=engine.apply_model(adapter.finish);assert len(counts)==1;item['selected_dispatch']=counts[0];item['arm']=mode;item['allow_bf16_reduced_precision_reduction']=precision_setting;item['allow_bf16_reduced_precision_reduction_split_k']=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k
   if diagnostic:item['public_logprobs']=public_logprobs(final_outputs,keys)
   return item
  result['phase']='gate_native_shadow';save()
  result['shadow_request']=request('gateshadow','shadow',True);save()
  assert native_finish_gate(result['shadow_request']['selected_dispatch']),'Native24 baseline HOLD; TileIR request not executed'
  result['phase']='gate_replacement';save()
  result['replacement_request']=request('gatereplace','replace',True);save()
  assert result['shadow_request']['cross_runtime_tokens_equal'] is True and result['replacement_request']['cross_runtime_tokens_equal'] is True
  result['public_logprob_gate']=compare_public(result['shadow_request']['public_logprobs'],result['replacement_request']['public_logprobs']);assert result['public_logprob_gate']['exact_numeric'] and result['public_logprob_gate']['values']==2560;save()
  result['phase']='gate_correctness_complete';policy.restore()
  result['adapter_restore']=engine.apply_model(adapter.close);adapter_installed=False
  torch.cuda.synchronize();close_attempted=True;engine.engine_core.shutdown();result['engine_shutdown']=True;engine=None
  result.update(complete=True,phase='complete');save()
 except BaseException as e:
  result.update(error=repr(e),complete=False);save();raise
 finally:
  final_cleanup(result,save,policy,engine,adapter,adapter_installed,close_attempted,
                lambda: __import__('torch').cuda.synchronize(),lambda: signal.alarm(0),
                propagating=sys.exc_info()[0] is not None)
if __name__=='__main__':main()
