"""Bounded best-effort cleanup retaining UVA owners through synchronization."""
def release_checked(model,adapter):
 adapter.release(model);return not hasattr(model,adapter.ATTR)
def final_cleanup(result,save,policy,engine,adapter,installed,sync,cancel_alarm,*,propagating):
 from functools import partial
 errors=[]
 def attempt(stage,fn):
  try:return True,fn()
  except BaseException as e:errors.append(dict(stage=stage,error=repr(e)));return False,None
 if policy is not None:attempt('precision_readonly_check',policy.restore)
 if engine is not None:
  a,_=attempt('synchronize_before_close',sync);result['synchronized_before_close']=a
  closed=not installed
  if installed:
   closed,r=attempt('adapter_close',lambda:engine.apply_model(adapter.close))
   if closed:result['adapter_restore']=r
  b,_=attempt('synchronize_after_close',sync);result['synchronized_after_close']=b
  if installed and a and b and closed:
   released,r=attempt('adapter_release',lambda:engine.apply_model(partial(release_checked,adapter=adapter)))
   if released:
    result['adapter_release_absent']=r;result['adapter_released']=r==[True]
    if not result['adapter_released']:errors.append(dict(stage='adapter_release_absence',error=repr(r)))
  ok,_=attempt('engine_shutdown',engine.engine_core.shutdown);result['engine_shutdown']=ok
 attempt('cancel_alarm',cancel_alarm)
 result['complete']=bool(not propagating and not errors and result.get('candidate_checks_passed'))
 if errors:result['cleanup_errors']=errors
 if result['complete']:result['phase']='complete'
 ok,_=attempt('save',save)
 if not ok:result['complete']=False
 if errors and not propagating:raise RuntimeError('Cleanup HOLD: '+repr(errors))
