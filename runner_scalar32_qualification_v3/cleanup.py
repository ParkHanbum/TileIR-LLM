def cleanup(result,save,policy,engine,observer,sync,cancel_alarm,propagating):
 errors=[]
 def attempt(stage,fn):
  try:fn();return True
  except BaseException as e:errors.append({'stage':stage,'type':type(e).__name__});return False
 if policy is not None:attempt('precision',policy.restore)
 result['observer_restored']=True if observer is None else attempt('restore',observer.close)
 result['synchronized_after_restore']=attempt('sync',sync) if engine is not None else True
 result['engine_shutdown']=attempt('shutdown',engine.engine_core.shutdown) if engine is not None else True
 attempt('cancel_alarm',cancel_alarm)
 result['complete']=bool(not propagating and not errors and result.get('candidate_checks_passed'))
 if errors:result['cleanup_errors']=errors
 if result['complete']:result['phase']='complete'
 save()
 if errors and not propagating:raise RuntimeError('cleanup failure')
