"""Restore public bindings, sync and shutdown before releasing retained owners."""
def cleanup(result,save,policy,engine,observer,sync,cancel_alarm,propagating):
 errors=[]
 def attempt(stage,fn):
  try:fn();return True
  except BaseException:errors.append({'stage':stage});return False
 if policy is not None:attempt('precision_readonly_check',policy.restore)
 restored=True if observer is None else attempt('restore',observer.close)
 result['observer_restored']=restored
 synced=attempt('synchronize_after_restore',sync)if observer is not None else True
 result['synchronized_after_restore']=synced
 shutdown=attempt('engine_shutdown',engine.engine_core.shutdown)if engine is not None else True
 result['engine_shutdown']=shutdown
 # Model may fail after observer install. Always attempt restoration and sync;
 # never drop owners when any cleanup failed.
 if observer is not None and restored and synced and shutdown and not errors:
  attempt('owner_release',observer.release_after_sync)
 if observer is not None:result['ownership']=observer.safe_receipt()
 attempt('cancel_alarm',cancel_alarm)
 result['complete']=bool(not propagating and not errors and result.get('candidate_checks_passed'))
 if errors:result['cleanup_errors']=errors
 if result['complete']:result['phase']='complete'
 save()
 if errors and not propagating:raise RuntimeError('ownership cleanup failed; retained refs')
