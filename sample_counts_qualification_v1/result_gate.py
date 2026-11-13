def result_gate(r):
 rows=r.get('records',[]);complete=r.get('complete') is True and r.get('gpu_executed') is True and not r.get('error') and len(rows)==4 and {(x.get('phase'),x.get('arm')) for x in rows}=={(p,a) for p in ('prefill','decode') for a in ('native','tileir')}
 passed=complete
 for row in rows:
  passed=passed and row.get('mapped_owner_pinned') is True and row.get('mapped_pointer_equal') is True and len(row.get('repeats',[]))==2
  for rep in row.get('repeats',[]):
   passed=passed and rep.get('call_count')==1 and rep.get('full_storage_raw_equal')==[True,True] and all(rep.get(k) is True for k in ('input_unchanged','first_return_same_object','rejected_disjoint','output_layout_equal','repeat_raw_equal'))
 return dict(complete=bool(complete),raw_pass=bool(passed),raw_difference_hold=bool(r.get('raw_difference_hold') is True or (complete and not passed)))
