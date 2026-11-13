def result_gate(r):
 rows=r.get('records',[])
 complete=r.get('complete') is True and r.get('gpu_executed') is True and not r.get('error') and len(rows)==4 and {(x.get('phase'),x.get('placement')) for x in rows}=={(p,d) for p in ('prefill','decode') for d in ('mapped_uva_primary','device_copy_control')}
 valid=complete
 for row in rows:
  reps=row.get('repeats',[])
  valid=valid and len(reps)==2
  if row.get('placement')=='mapped_uva_primary':valid=valid and row.get('mapped_owner_pinned') is True and row.get('mapped_pointer_equal') is True
  for rep in reps:valid=valid and rep.get('input_unchanged') is True and rep.get('views_alias_expected') is True and rep.get('repeat_raw_equal') is True and rep.get('all_group_storage_raw_equal')==[True]*4
 return dict(complete=bool(complete),raw_pass=bool(valid),raw_difference_hold=bool(complete and not valid))
