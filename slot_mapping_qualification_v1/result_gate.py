def result_gate(r):
 rows=r.get('records',[])
 complete=r.get('complete') is True and r.get('gpu_executed') is True and not r.get('error') and len(rows)==2 and {x.get('phase') for x in rows}=={'prefill','decode'}
 passed=complete
 for row in rows:
  reps=row.get('repeats',[]);passed=passed and len(reps)==2
  for rep in reps:
   passed=passed and all(rep.get(k) is True for k in ('full_storage_raw_equal','input_unchanged','views_alias_expected','full_capacity_pad','repeat_raw_equal'))
 return dict(complete=bool(complete),raw_pass=bool(passed),raw_difference_hold=bool(complete and not passed))
