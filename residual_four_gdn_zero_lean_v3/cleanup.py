"""Restore zero hooks before four/base cleanup; retain both owners until safe sync."""
def finish_all(result,save,hooks,four_cleanup,*,propagating):
    errors=[];restored=hooks is None
    if hooks is not None:
        try:hooks.close();restored=True
        except BaseException:errors.append('zero_restore')
    if errors:result['candidate_checks_passed']=False
    finished=False
    try:four_cleanup(propagating or bool(errors));finished=True
    finally:
        if hooks is not None:
            if restored and finished and not errors and not result.get('cleanup_errors') and not result.get('four_cleanup_errors') and result.get('synchronized_before_close') and result.get('synchronized_after_close') and result.get('engine_shutdown'):
                try:hooks.release(after_synchronize=True)
                except BaseException:errors.append('zero_release')
            result['zero_lean']=hooks.receipt()
        if errors:result.update(complete=False,zero_cleanup_errors=errors)
        save()
    if errors and not propagating:raise RuntimeError('combined zero cleanup HOLD')
