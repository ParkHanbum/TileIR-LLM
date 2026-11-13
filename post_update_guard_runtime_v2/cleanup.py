from residual_four_lean26_runtime_v1.cleanup import close_and_finish

def finish(result,save,checks,owners,replacement,final,*,propagating):
 errors=[]
 if checks is not None:
  try:checks.binding.close();result['post_guard_binding_restored']=True
  except BaseException as e:errors.append(type(e).__name__)
 if errors:result.update(candidate_checks_passed=False,post_guard_cleanup_errors=errors)
 try:close_and_finish(result,save,owners,replacement,final,propagating=propagating or bool(errors))
 finally:
  if errors:result['complete']=False;save()
 if errors and not propagating:raise RuntimeError('Post-update guard binding restoration failed')
