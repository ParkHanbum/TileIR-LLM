"""CPU-default plan; parent-reviewed native-to-lean correctness only, no timing."""
import argparse,json,os,sys,signal
from pathlib import Path
from functools import partial
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
if '--run-gpu' in sys.argv:
 from whole_v7_public_blas_logger_v4.gates import require_ready
 require_ready()
if __name__=='__main__' and '--run-gpu' not in sys.argv:
 print(json.dumps(dict(executed=False,framework_imported=False,scope='Whole combined endpoint CPU-ready; four new raw-qualified lean domains pending together')));raise SystemExit(0)
from whole_lean_extensions_runtime_v7 import adapter
from lean_public_runtime_qualification_v2.cleanup import final_cleanup
from whole_lean_extensions_runtime_v7.gates import sha,frozen_reference,exact_tokens,exact_public,dispatch
from all_gemm_sixteen_sampler_composition_v2.cache_gate import preflight
from all_gemm_sixteen_sampler_composition_v2.precision import PrecisionPolicy
from packed_vllm_runtime_compare_v4.gates import public_logprobs
SOURCE_ROOT=Path(__file__).resolve().parent;P=SOURCE_ROOT.parent
R=Path(os.environ.get('PUBLIC_BLAS_AUDIT_ROOT',str(SOURCE_ROOT)))
if '--run-gpu' in sys.argv:
 assert R.parent==P and R.name in tuple(SOURCE_ROOT.name+'_process'+str(i) for i in range(3))
PYTHON='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
IDS='/home/n/Documents/Codex/2026-10-31/task/tileir_poc/near40-model-search-20261109/g16-balanced-continuation-20261109/inputs/input-ids.json'
LOCK='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock'
from whole_v7_public_blas_logger_v4.gates import qualification
def main():
 from whole_lean_extensions_runtime_v7.preflight import source_preflight
 source_preflight(R)
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-gpu',action='store_true');ap.add_argument('--output',type=Path);a=ap.parse_args()
 ids=json.loads(Path(IDS).read_text());assert len(ids)==53 and sha(IDS)=='2843e98c58086b11e16679025ea499e97c360b2c4853ff0c211228ace09694dd'
 frozen_tokens,frozen_public,options=frozen_reference()
 result=dict(complete=False,gpu_executed=False,input_ids=ids,input_sha256=sha(IDS),engine_options=options,scope='current-eager native-to-lean correctness only',timed_requests=0,expected_boundaries=5889,expected_logical_operations=6057,proof_scope='New lean endpoint/census/operational safety gates; previous combined raw proof is referenced, not repeated.',baseline='Current eager vLLM mode0/graphsNONE B32P53G16 logprobs5 T/T; not historical optimized baseline',native_residuals=['retained public API and outer scope guards','remaining unclassified runner/output operations'],validation_scope='No per-call payload oracle/full-backing D2H; operational value guards and UVA owner lifetime retained; endpoint qualification only')
 if not a.run_gpu:print(json.dumps(result,indent=2));return
 result['qualification_receipt']=qualification(R) # First actual-run gate, before lock/import/model.
 assert os.environ.get('VLLM_COMPARISON_PARENT_APPROVED')=='1'and Path(sys.executable).absolute()==Path(PYTHON)
 assert a.output and a.output.is_absolute()and not a.output.exists()
 for p,h in json.loads((R/'PINS.json').read_text()).items():assert sha(p)==h,p
 held=os.fstat(int(os.environ['GEMM_COMPOSED_LOCK_FD']));target=Path(LOCK).stat();assert(held.st_dev,held.st_ino)==(target.st_dev,target.st_ino)
 result['qualification_receipt']=qualification(R);result['cache_preseed']=preflight(R)
 a.output.mkdir()
 def save():(a.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 def timeout(*_):raise TimeoutError('Bounded900s child; parent owns cleanup/recovery')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_ENABLE_V1_MULTIPROCESSING='0',DO_NOT_TRACK='1')
 result['runtime_environment']={k:v for k,v in os.environ.items()if k.startswith(('VLLM_','TORCHINDUCTOR_','TRITON_','FLASHINFER_','CUDA_'))or k in('XDG_CACHE_HOME','LD_LIBRARY_PATH','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE')}
 engine=None;policy=None;installed=False;owners=None;replacement=None;zero_hooks=None;ext=None
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
  from residual_four_lean26_runtime_v1.ownership import Owners
  owners=Owners();owners.install()
  from residual_four_gdn_zero_lean_v3.zero_hooks import Hooks
  import vllm.v1.worker.gpu.model_runner as mr
  import vllm.v1.worker.utils as u
  zero_hooks=Hooks();zero_hooks.install(mr,u);adapter.ZERO=zero_hooks
  from whole_lean_extensions_runtime_v7.extension import Extension
  ext=Extension();ext.pre_engine(torch);adapter.EXT=ext;adapter.OUTPUT=a.output/'integer-pair'
  result['method_bindings']=ext.method_bindings
  engine=LLMEngine.from_engine_args(EngineArgs(**options),usage_context=UsageContext.LLM_CLASS,enable_multiprocessing=False)
  torch.cuda.synchronize();cfg=engine.vllm_config
  assert type(engine.engine_core).__name__=='InprocClient'and cfg.model_config.enforce_eager
  assert int(cfg.compilation_config.mode)==0 and cfg.compilation_config.cudagraph_mode.name=='NONE'and cfg.scheduler_config.async_scheduling is True
  result['resolved_config']={k:str(getattr(cfg,k+'_config'))for k in('model','scheduler','cache','compilation','parallel')};result['resolved_async_scheduling']=cfg.scheduler_config.async_scheduling
  params=SamplingParams(n=1,temperature=0,min_tokens=16,max_tokens=16,ignore_eos=False,detokenize=False,logprobs=5,prompt_logprobs=None,seed=20261111,output_kind=RequestOutputKind.CUMULATIVE)
  def request(mode):
      with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=False, with_stack=False, profile_memory=False, with_flops=False) as _request_prof:
          with torch.profiler.record_function('request_install::' + mode):
              assert mode in ('native', 'lean') and (not engine.has_unfinished_requests())
              _, precision = policy.select('native')
              if mode == 'lean':
                  engine.apply_model(partial(adapter.start, mode='lean'))
              keys = [mode + ':' + str(i) for i in range(32)]
              outputs = {}
              torch.cuda.synchronize()
              torch.cuda.reset_peak_memory_stats()
          from whole_v7_public_blas_logger_v4.logger import Window
          with Window(a.output, mode) as _log:
              with torch.profiler.record_function('request_execution::' + mode):
                  for key in keys:
                      engine.add_request(key, {'prompt_token_ids': list(ids)}, params)
                  while engine.has_unfinished_requests():
                      for o in engine.step():
                          if o.finished:
                              assert o.request_id in keys and o.request_id not in outputs and (len(o.outputs) == 1)
                              outputs[o.request_id] = o
                  torch.cuda.synchronize()
                  assert set(outputs) == set(keys)
          result[mode + '_logger'] = _log.verify()
          save()
          with torch.profiler.record_function('request_proof::' + mode):
              tokens = [list(outputs[k].outputs[0].token_ids) for k in keys]
              public = public_logprobs(outputs, keys)
              item = dict(mode=mode, output_token_ids=tokens, public_logprobs=public, selected_dispatch=None, allow_bf16_reduced_precision_reduction=precision, allow_bf16_reduced_precision_reduction_split_k=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k, peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
              result[mode + '_request'] = item
              save()
              item['frozen_tokens_gate'] = exact_tokens(tokens, frozen_tokens)
              item['frozen_public_gate'] = exact_public(public, frozen_public)
              if mode == 'lean':
                  d = engine.apply_model(adapter.finish)
                  assert len(d) == 1
                  item['selected_dispatch'] = d[0]
                  item['dispatch_gate'] = dispatch(d[0])
              save()
          with torch.profiler.record_function('request_restore::' + mode):
              pass
      _trace_path = a.output / (mode + '.trace.json')
      _request_prof.export_chrome_trace(str(_trace_path))
      from whole_lean_profile_prepare_v4.runtime import summarize_trace
      _profile_summary = summarize_trace(_trace_path, mode)
      item['profile_partition'] = _profile_summary
      item['profiled'] = True
      item['whole_ms'] = None
      save()
      assert _profile_summary['denominator_ready'], 'Unresolved request kernel denominator'
      return item
  installed=True # Pair partial installation also needs adapter cleanup if native reference fails.
  result['pair_reference_install']=engine.apply_model(adapter.install_reference)
  engine.apply_model(adapter.start_reference)
  zero_hooks.begin_native_reference();result['phase']='native_reference';save();native=request('native')
  observed=engine.apply_model(adapter.finish_reference);assert len(observed)==1
  result['native_request']['pair_observation']=observed[0];save()
  assert not engine.has_unfinished_requests();torch.cuda.synchronize()
  from residual_four_gdn_owned_lean_v2.four import Replacement
  replacement=Replacement(a.output/'four-replacement',owners);replacement.install()
  result['phase']='install_lean';save();installed=True
  result['adapter_install']=engine.apply_model(adapter.install);assert len(result['adapter_install'])==1
  ext.post_base(engine)
  ext.install_post_guard(replacement)
  result['output_copy_install']=ext.install_output_copy()
  result['new_domain_bindings']=dict(scalar_owner_verified=True,metadata_owned_callbacks=True,metadata_batch_joint_active=ext.metadata_joint.active(),integer_installed_before_base=True)
  from residual_four_lean26_runtime_v1.adapter import bind_lean_controller
  result['four_binding']=engine.apply_model(bind_lean_controller);assert result['four_binding']==[dict(shared_latch=True,owned_copy_identity=True,per_helper_cap=16)]
  result['phase']='lean_correctness';save();replacement.active=True;lean=request('lean');replacement.active=False
  result['four_replacement']=replacement.finish();result['zero_lean']=zero_hooks.receipt();save()
  result['same_run_token_gate']=exact_tokens(native['output_token_ids'],lean['output_token_ids'])
  result['same_run_public_gate']=exact_public(native['public_logprobs'],lean['public_logprobs'])
  result.update(candidate_checks_passed=True,phase='candidate_checks_complete');save()
 except BaseException as e:
  from whole_lean_extensions_runtime_v7.failure import snapshot
  result['first_failure']=snapshot(ext)
  result.update(error=repr(e),complete=False,failure_type=type(e).__name__,automatic_retry=False,runtime_settings_changed=False);save();raise
 finally:
  from residual_four_gdn_zero_lean_v3.cleanup import finish_all
  from residual_four_lean26_runtime_v1.cleanup import close_and_finish
  def four_cleanup(propagating):
   close_and_finish(result,save,owners,replacement,
    lambda propagating:final_cleanup(result,save,policy,engine,adapter,installed,lambda:ext.final_sync(__import__('torch').cuda.synchronize,result) if ext is not None else __import__('torch').cuda.synchronize(),lambda:signal.alarm(0),propagating=propagating),
    propagating=propagating)
  extension_errors=[]
  if ext is not None:
   try:ext.close_post();result['extensions_restored']=True
   except BaseException:extension_errors.append('extension_restore');result['candidate_checks_passed']=False
  try:finish_all(result,save,zero_hooks,four_cleanup,propagating=sys.exc_info()[0]is not None or bool(extension_errors))
  finally:
   if ext is not None and not extension_errors and result.get('complete'):
    try:ext.release_after_shutdown(result)
    except BaseException:extension_errors.append('extension_release')
   if extension_errors:result.update(complete=False,extension_cleanup_errors=extension_errors)
   save()
  if extension_errors:raise RuntimeError('Whole extension cleanup HOLD')
if __name__=='__main__':main()
