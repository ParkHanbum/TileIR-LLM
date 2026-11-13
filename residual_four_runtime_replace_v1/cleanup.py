"""Keep UVA owners alive through final synchronization and engine cleanup."""
def close_and_finish(result,save,owners,replacement,final,*,propagating):
 errors=[]
 def attempt(stage,fn):
  try:fn()
  except BaseException as e:errors.append(dict(stage=stage,error=repr(e)))
 if replacement is not None:
  attempt('helper_close',replacement.close)
  def record():result['four_replacement']=replacement.summary();replacement.save()
  attempt('helper_record',record)
 if owners is not None:
  attempt('constructor_restore',owners.close);result['uva_constructor_restored']=owners.restored
 if errors:result.update(candidate_checks_passed=False,four_cleanup_errors=errors)
 final_ok=False
 try:final(propagating or bool(errors));final_ok=True
 finally:
  if final_ok and owners is not None and not errors and not result.get('cleanup_errors') and (replacement is None or replacement.restored) and result.get('synchronized_before_close') and result.get('synchronized_after_close') and result.get('engine_shutdown') and owners.restored:
   def release():result['uva_release']=owners.release(True)
   attempt('owner_release',release)
  if errors:result.update(complete=False,four_cleanup_errors=errors)
  attempt('save',save)
 if errors and not propagating:raise RuntimeError('Four-helper cleanup HOLD: '+repr(errors))
