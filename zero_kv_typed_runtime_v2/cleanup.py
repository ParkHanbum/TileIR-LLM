"""Keep strong owner refs if any mandatory teardown stage fails."""
def teardown(hooks,policy,engine,synchronize,result):
 def do(label,fn):
  try:fn();result[label]=True;return True
  except BaseException:result['cleanup_errors'].append(label);return False
 restored=do('hooks_restored',hooks.close)
 synced=do('synchronized',synchronize)
 precision=True if policy is None else do('precision_restored',policy.restore)
 shutdown=False if engine is None else do('engine_shutdown',engine.engine_core.shutdown)
 if restored and synced and precision and shutdown:
  do('owners_released',lambda:hooks.release(after_synchronize=True))
 else:result['owners_retained']=True
