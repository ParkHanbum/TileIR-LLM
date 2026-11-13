"""CPU-only exact census/source-launch accounting. No physical kernel claim."""
SCOPE='source ct.launch boundary; not physical GPU count'
def check_stream(before,after):
 assert before==after,'Candidate changed current stream'
 return True
def check_ledger(row):
 typed=row['typed_nonempty'];ledger=row['launch_ledger']
 assert 1<=len(typed)<=4 and len(ledger)==len(typed)
 assert all(type(x)is bool for x in typed)and row['nonempty']==any(typed)
 if row['label']=='staged':assert len(typed)==1
 for active,entry in zip(typed,ledger):
  assert entry['nonempty']is active and entry['scope']==SCOPE
  assert type(entry['launch_attempts'])is int and type(entry['launch_returns'])is int
  assert entry['launch_attempts']==entry['launch_returns']==int(active)
 assert row['candidate_stream_unchanged']is True
 return True
def check_rows(rows,counts):
 assert len(rows)==140
 for label,total,nonempty in [('staged',123,5),('fused',17,1)]:
  selected=[r for r in rows if r['label']==label]
  assert len(selected)==total and all(type(x['nonempty'])is bool for x in selected)
  assert sum(x['nonempty']for x in selected)==nonempty
  assert counts['request.'+label+'.nonempty']==nonempty and counts['request.'+label+'.empty']==total-nonempty
 for row in rows:check_ledger(row)
 return True
