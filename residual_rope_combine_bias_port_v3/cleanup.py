"""Reverse helper restoration; errors must not skip remaining model cleanup."""
def close_helpers(result,post_context,post_shadow,shadows):
 errors=[]
 if post_shadow is not None:
  try:result['post_update_shadow']=post_shadow.snapshot()
  except BaseException as e:errors.append(dict(stage='post_snapshot',error=repr(e)))
 if post_context is not None:
  try:post_context.__exit__(None,None,None);result['post_update_restored']=True
  except BaseException as e:result['post_update_restored']=False;errors.append(dict(stage='post_close',error=repr(e)))
 if shadows is not None:
  try:shadows.close()
  except BaseException as e:errors.append(dict(stage='three_close',error=repr(e)))
  try:result['residual_shadow']=shadows.summary();shadows.save()
  except BaseException as e:errors.append(dict(stage='three_save',error=repr(e)))
 if errors:result.update(candidate_checks_passed=False,shadow_cleanup_error=errors)
 return errors
