"""Narrow new-domain lifecycle around existing combined lean runner."""
from .bridge import Bus
class Extension:
 def __init__(self):
  self.model=None;self.controller=None;self.phase='setup';self.bus=Bus(lambda:self.controller);self.staged=None;self.scalar=None;self.metadata=None;self.pair=None;self.post_restored=False;self.staged_restored=False;self.request_started=False;self.request_finished=False;self.retained=[];self.reference_observed=None;self.metadata_joint=None;self.batch_cast=None;self.post_guard=None;self.ownership_pair=None;self.output_copy=None;self.copy_sync_failed=False;self.copy_sync_receipt=None;self.copy_restored=False;self.pair_restore=None
 def pre_engine(self,torch):
  from staged_typed_lean_runtime_v1.replacement import OwnershipObserver
  from staged_ownership_runtime_v1.observer import comparator
  from staged_typed_lean_runtime_v1.source_gate import verify
  import vllm.v1.worker.gpu.buffer_utils as bm
  import vllm.v1.worker.gpu.block_table as bt
  from pathlib import Path
  self.method_bindings=verify(bm,bt,Path(__file__).resolve().parent.parent/'staged_typed_lean_runtime_v1')
  self.staged=OwnershipObserver(bm,bt,phase=lambda:self.phase,compare_entries=comparator(torch),synchronize=torch.cuda.synchronize,current_stream=torch.cuda.current_stream)
  self.staged.bind_controller(self.bus);self.staged.__enter__();self.phase='warmup'
  from ownership_table_pair_equal_lean_v1.binding import Binding as OwnershipPair
  self.ownership_pair=OwnershipPair(self.staged,torch,lambda:self.phase)
  self.ownership_pair.install();self.retained.append(self.ownership_pair)
 def pair_before_base(self,model,output):
  from integer_pair_runtime_v1 import adapter as pair
  self.model=model
  try:
   result=pair.install_pair(model,output);self.pair=getattr(model,pair.PAIR);self.retained.extend([self.pair,self.pair.cast,self.pair.integer]);return result
  except BaseException as e:
   self.pair=getattr(model,pair.PAIR,None);self.bus.latch('pair/install',e);raise
 def reference_start(self):
  self.bus.guard()
  if self.phase!='warmup' or self.pair is None or self.reference_observed is not None:raise RuntimeError('Invalid native observation phase')
  self.pair.start('shadow')
 def reference_finish(self):
  self.bus.guard()
  if self.reference_observed is not None:raise RuntimeError('Repeated native observation finish')
  self.reference_observed=self.pair.finish()
  return dict(current_run_native_observed=True,current_run_raw_comparison=False,integer_pair=self.reference_observed)
 def request_guard(self):
  self.bus.guard()
  if self.phase!='request' or not self.request_started or self.request_finished:raise RuntimeError('New-domain entry outside request')
 def request_entry(self,label,fn):
  def wrapped(*a,**kw):
   try:self.request_guard();return fn(*a,**kw)
   except BaseException as e:self.bus.latch(label,e);raise
  return wrapped
 def post_base(self,engine):
  from lean26_gdn_finite_runtime_v1.adapter import ATTR
  self.controller=getattr(self.model,ATTR);self.bus.guard()
  from runner_scalar32_lean_v1.controller import Controller as Scalar
  from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
  from vllm.v1.worker.gpu.model_runner import GPUModelRunner
  from runner_scalar32_qualification_v3.caller_gate import verify
  runner=engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
  assert type(runner)is GPUModelRunner and type(runner.model_state)is MambaHybridModelState and engine.vllm_config.speculative_config is None
  self.caller_source=verify(runner);self.scalar=Scalar(runner.model_state);self.bus.bind_scalar(self.scalar);self.scalar.check_global=self.request_guard
  self.scalar.binding.replacement=self.request_entry('scalar/entry',self.scalar.binding.replacement)
  self.bus.install('scalar',self.scalar.install,self.scalar.close)
  from runtime_metadata_five_lean_v1.controller import Controller as Metadata
  from vllm.v1.attention.backend import CommonAttentionMetadata
  from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder,compute_causal_conv1d_metadata
  self.metadata=Metadata();self.bus.bind_metadata(self.metadata)
  old_guard=self.metadata.guard
  self.metadata.guard=lambda fn:self.request_entry('metadata/entry',old_guard(fn))
  from batch_ptr_int_cast_lean_v1.controller import Controller as BatchCast
  from metadata_batch_cast_bridge_v1.bridge import Bridge as Joint
  self.batch_cast=BatchCast();self.metadata_joint=Joint(self.metadata,self.batch_cast)
  # Outer entry guard covers pre-validation errors and both fresh copied class methods.
  old_joint_guard=self.metadata_joint.guard
  self.metadata_joint.guard=lambda fn:self.request_entry('metadata_batch/entry',old_joint_guard(fn))
  self.bus.install('metadata_batch',lambda:self.metadata_joint.install(CommonAttentionMetadata,GDNAttentionMetadataBuilder,compute_causal_conv1d_metadata),self.metadata_joint.close)
  self.retained.extend([runner,self.scalar,self.metadata,self.batch_cast,self.metadata_joint])
 def install_output_copy(self):
  import functools
  from gdn_prefill_output_copy_lean_v2.adapter import Adapter as OutputCopy
  self.bus.guard()
  if self.output_copy is not None:raise RuntimeError('Repeated output copy install')
  self.output_copy=OutputCopy();self.retained.append(self.output_copy)
  try:
   result=self.output_copy.install(self.model)
   for i,(m,had,previous,old) in enumerate(self.output_copy.bindings):
    if vars(m).get('_forward_core')is not old:raise RuntimeError('Copy install identity')
    guarded=functools.wraps(old)(self.request_entry('output_copy/entry',old))
    setattr(m,'_forward_core',guarded)
    self.output_copy.bindings[i]=(m,had,previous,guarded)
   return result
  except BaseException as e:self.bus.latch('output_copy/install',e);raise
 def final_sync(self,sync,result):
  if self.output_copy is None or self.copy_sync_receipt is not None or self.copy_sync_failed:return sync()
  # First sync is unchanged. Release only at existing post-close sync after all
  # restoration stages so far succeeded; any failure retains copy owners.
  safe=(self.copy_restored and self.post_restored and self.staged_restored and self.pair_restore==dict(restored=True,original_callback_identity=True) and result.get('candidate_checks_passed')is True and result.get('synchronized_before_close')is True and bool(result.get('adapter_restore')) and not any(result.get(k)for k in ('cleanup_errors','four_cleanup_errors','zero_cleanup_errors','extension_cleanup_errors')))
  if not safe:return sync()
  try:
   self.copy_sync_receipt=self.output_copy.synchronize_and_release(sync)
   result['output_copy_sync']=self.copy_sync_receipt
  except BaseException:
   self.copy_sync_failed=True
   raise
 def install_post_guard(self,owner):
  from post_update_guard_lean_v1.adapter import Checks
  self.bus.guard()
  if self.post_guard is not None:raise RuntimeError('Repeated post guard install')
  self.post_guard=Checks(owner)
  self.post_guard.binding.callback=self.request_entry('post_guard/entry',self.post_guard.validate)
  # Attempt registered first, so partial install still receives cleanup.
  self.bus.install('post_guard',self.post_guard.binding.install,self.post_guard.binding.close)
  self.retained.extend([owner,self.post_guard])
 def start_new(self):
  self.bus.guard()
  if self.reference_observed is None:raise RuntimeError('Native pair observation required')
  if self.request_started or self.phase!='warmup':raise RuntimeError('Repeated request')
  self.pair.start('replace');self.scalar.start();self.metadata_joint.start();self.post_guard.start();self.request_started=True;self.phase='request'
 def finish_new(self):
  self.request_guard()
  try:
   post=self.post_guard.finish();joint=self.metadata_joint.finish()
   result=dict(output_copy=self.output_copy.finish(),ownership_pair=self.ownership_pair.finish(),post_guard=post,integer_pair=self.pair.finish(),scalar=self.scalar.finish(),metadata=joint['metadata'],batch_cast=joint['batch_cast'],metadata_batch_joint=joint,staged=self.staged.safe_receipt())
   self.request_finished=True;self.phase='finished';return result
  except BaseException as e:self.bus.latch('new_domains/finish',e);raise
 def close_post(self):
  self.phase='cleanup';errors=[]
  # Copy core wrapper must be removed before EQ/core restoration.
  if self.output_copy is not None:
   try:self.output_copy.close();self.copy_restored=True
   except BaseException as e:errors.append(e)
  # Each independent restoration is attempted; any failure forbids owner release.
  try:self.bus.close();self.post_restored=True
  except BaseException as e:errors.append(e)
  if self.ownership_pair is not None:
   try:self.pair_restore=self.ownership_pair.close()
   except BaseException as e:errors.append(e)
  if self.staged is not None:
   try:self.staged.close();self.staged_restored=True
   except BaseException as e:errors.append(e)
  if errors:raise RuntimeError('Extension reverse restoration HOLD') from errors[0]
 def release_after_shutdown(self,result):
  self.bus.require_release(synchronized=result.get('synchronized_after_close')is True,shutdown=result.get('engine_shutdown')is True)
  if not self.post_restored or not self.staged_restored or result.get('cleanup_errors')or result.get('four_cleanup_errors')or result.get('zero_cleanup_errors'):raise RuntimeError('Retain extension owners')
  if self.output_copy is not None and (not self.copy_restored or self.copy_sync_failed or self.copy_sync_receipt is None or self.output_copy.pending):raise RuntimeError('Retain output copy owners')
  if self.ownership_pair is not None and self.pair_restore!=dict(restored=True,original_callback_identity=True):raise RuntimeError('Retain pair owners')
  result['ownership_pair_restore']=self.pair_restore;result['output_copy_restored']=self.copy_restored
  self.staged.release_after_sync();result['staged_ownership']=self.staged.safe_receipt();self.retained.clear();result['extension_owners_released']=True
