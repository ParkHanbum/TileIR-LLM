"""Exact launch-correlated API/symbol mapping; no tile-size or ordinal inference."""
def attribute(events,mode,verified_symbols):
 execution=[e for e in events if e.get('ph')=='X' and e.get('cat')=='user_annotation' and e.get('name')=='request_execution::'+mode]
 assert len(execution)==1
 r=execution[0]
 def contains(a,b):return a['pid']==b.get('pid') and a['tid']==b.get('tid') and a['ts']<=b['ts'] and b['ts']+b['dur']<=a['ts']+a['dur']
 hosts={}
 for e in events:
  if e.get('ph')=='X' and e.get('cat')in('cuda_runtime','cuda_driver'):
   c=e.get('args',{}).get('correlation')
   if type(c)is int and c>0:hosts.setdefault(c,[]).append(e)
 ranges=[e for e in events if e.get('ph')=='X' and e.get('cat')=='user_annotation' and e.get('name','').startswith(('tileir_public_api::','native_public_api::','lean_public_api::'))]
 rows=[]
 for e in events:
  if e.get('ph')!='X' or e.get('cat')!='kernel':continue
  hs=hosts.get(e.get('args',{}).get('correlation'),[])
  if not hs or not all(contains(r,h)for h in hs):continue
  per_host=[]
  for h in hs:
   scopes=[s for s in ranges if contains(s,h)]
   per_host.append(min(scopes,key=lambda s:s['dur'])['name']if scopes else None)
  name=per_host[0]if len(set(per_host))==1 else None
  sym=verified_symbols.get(e.get('name')) # FULL exact actual symbol only.
  own=bool(sym and sym.get('verified_own_elf')is True and sym.get('artifact_sha256') and sym.get('source_file') and sym.get('entry'))
  rows.append(dict(kernel=e.get('name'),correlation=e.get('args',{}).get('correlation'),api=name,duration_us=e['dur'],own_tileir_verified=own,symbol_evidence=sym if own else None))
 return dict(rows=rows,unattributed=sum(x['api']is None for x in rows),unverified_symbols=sum(not x['own_tileir_verified']for x in rows),scope='Only correlated request execution. Unknown symbols remain unknown; API wrappers may contain validators/native work.')
