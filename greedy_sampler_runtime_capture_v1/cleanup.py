"""Independent best-effort stages, adapted from composed v5 cleanup."""
def final_cleanup(result,save,check_flags,engine,observer,live,shutdown_attempted,sync,cancel_alarm,*,propagating):
 errors=[]
 def attempt(stage,fn):
  try:return fn()
  except BaseException as e:errors.append(dict(stage=stage,error=repr(e)))
 def restore():result['capture_cleanup']=engine.apply_model(observer.close_only)
 def shutdown():engine.engine_core.shutdown();result['engine_shutdown']=True
 if check_flags is not None:attempt('precision_readonly_check',check_flags)
 if engine is not None:
  if live:attempt('observer_restore',restore)
  attempt('synchronize',sync)
  if not shutdown_attempted:attempt('engine_shutdown',shutdown)
 attempt('cancel_alarm',cancel_alarm)
 if errors:result.update(complete=False,cleanup_errors=errors)
 attempt('save',save)
 if errors:
  result.update(complete=False,cleanup_errors=errors)
  if not propagating:raise RuntimeError('Cleanup HOLD: '+repr(errors))
