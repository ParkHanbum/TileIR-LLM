"""Actual-owner asynchronous copy24. No oracle/private buffers/payload reads."""
from gdn_prefill_output_copy_v3.adapter import Adapter as ShadowAdapter
from gdn_prefill_output_copy_v3.api import check,metadata,SHAPE,BYTES
from .lifetime import Copy
class Adapter(ShadowAdapter):
 def install(self,model):
  receipt=super().install(model)
  receipt['scope']='Actual owned prefill output copy24 replacement; async retained owners'
  return receipt
 def observe(self,t,func,args,kwargs):
  ctx=self.context
  if str(func)!='aten.copy_.default' or not ctx:return func(*args,**kwargs)
  try:
   dst,src=args[:2];target=ctx['destination']
   if not isinstance(dst,t.Tensor)or not isinstance(src,t.Tensor):raise TypeError('Expected actual Tensor operands')
   if dst.untyped_storage().data_ptr()!=target.untyped_storage().data_ptr():return func(*args,**kwargs)
   if ctx['count']!=0 or dst.data_ptr()!=target.data_ptr()or tuple(dst.shape)!=SHAPE:raise RuntimeError('Unexpected destination copy')
   if len(args)>3 or (args[2]if len(args)>2 else kwargs.get('non_blocking',False)):raise RuntimeError('Unexpected copy options')
   check(dst,src,t)
   # Narrow to the descriptor actually raw-qualified in24 original calls.
   for x in(dst,src):
    if x.storage_offset()!=0 or x.untyped_storage().nbytes()!=BYTES:raise ValueError('Unqualified actual copy backing')
   ctx['count']+=1;stream=t.cuda.current_stream(dst.device).cuda_stream
   copier=Copy();self.pending.append(copier);ret=copier(dst,src)
   if ret is not dst or t.cuda.current_stream(dst.device).cuda_stream!=stream:raise RuntimeError('Copy identity/stream changed')
   def public(x):return {k:v for k,v in metadata(x).items()if k not in('ptr','storage_ptr')}
   self.rows.append(dict(layer=ctx['layer'],source=public(src),destination=public(dst),native_calls=0,candidate_calls=1,async_enqueued=True))
   return dst
  except BaseException:self.failed=True;raise
 def finish(self):
  super().finish()
  if len(self.pending)!=24 or any(c.failed or c.calls!=1 or len(c.pending)!=0 or not c.recorded for c in self.pending):raise RuntimeError('Copy enqueue census mismatch')
  return dict(complete=True,rows=self.rows,calls=24,model_copies_native=False,actual_replacement=True,pending_copy_owners=0,per_call_sync=False)
 def synchronize_and_release(self,synchronize):
  # Substitute this for the existing final device sync; do not call per op.
  # If synchronization raises, retain every pending reference.
  synchronize()
  for c in self.pending:c.release_after_sync()
  self.pending.clear()
  return dict(synchronized=True,copy_owners_released=True)
