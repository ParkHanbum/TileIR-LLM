"""Best-effort cleanup: all stages attempted; no precision flag writes."""
def final_cleanup(result,save,policy,engine,adapter,installed,shutdown_attempted,sync,cancel_alarm,*,propagating):
 errors=[]
 def attempt(stage,fn):
  try:return fn()
  except BaseException as e:
   errors.append(dict(stage=stage,error=repr(e)))
 def restore_adapter():result['adapter_restore']=engine.apply_model(adapter.close)
 def shutdown():
  engine.engine_core.shutdown();result['engine_shutdown']=True
 if policy is not None:attempt('precision_readonly_check',policy.restore)
 if engine is not None:
  attempt('synchronize',sync)
  if installed:attempt('adapter_close',restore_adapter)
  if not shutdown_attempted:attempt('engine_shutdown',shutdown)
 attempt('cancel_alarm',cancel_alarm)
 if errors:result.update(complete=False,cleanup_errors=errors)
 attempt('save',save)
 if errors:
  result.update(complete=False,cleanup_errors=errors)
  if not propagating:raise RuntimeError('Cleanup HOLD: '+repr(errors))
