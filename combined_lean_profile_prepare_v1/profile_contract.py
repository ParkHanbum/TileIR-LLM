"""CPU trace partition by CPU launch correlation, never by GPU timestamps alone."""
import json,math
from pathlib import Path

def scope_names(mode):
 assert mode in('native','lean')
 return dict(install='request_install::'+mode,execution='request_execution::'+mode,proof='request_proof::'+mode,restore='request_restore::'+mode)
def inspect_trace(path,mode):return inspect_events(json.loads(Path(path).read_text())['traceEvents'],mode)
def inspect_events(events,mode):
 names=scope_names(mode);ranges={}
 def interval(e):
  ts,dur=e.get('ts'),e.get('dur');assert isinstance(ts,(int,float))and isinstance(dur,(int,float))and math.isfinite(ts)and math.isfinite(dur)and dur>=0
  return ts,ts+dur
 for key,name in names.items():
  found=[e for e in events if e.get('ph')=='X'and e.get('cat')=='user_annotation'and e.get('name')==name]
  assert len(found)==1,('Exactly one explicit request phase range required',name,len(found))
  e=found[0];a,b=interval(e);assert b>a and 'pid'in e and 'tid'in e;ranges[key]=e
 assert len({(e['pid'],e['tid'])for e in ranges.values()})==1,'Phase range thread mismatch'
 ordered=[ranges[k]for k in('install','execution','proof','restore')]
 assert all(interval(a)[1]<=interval(b)[0]for a,b in zip(ordered,ordered[1:])),'Request phase ranges overlap or reorder'
 def phase(e):
  a,b=interval(e);matches=[]
  for key,r in ranges.items():
   ra,rb=interval(r)
   if(e.get('pid'),e.get('tid'))==(r['pid'],r['tid'])and ra<=a and b<=rb:matches.append(key)
  assert len(matches)<=1
  return matches[0]if matches else'outside'
 # A runtime+driver nested pair can share an ID. Accept only agreement of all correlated host records.
 hosts={}
 for e in events:
  if e.get('ph')=='X'and e.get('cat')in('cuda_runtime','cuda_driver'):
   c=e.get('args',{}).get('correlation')
   if isinstance(c,int)and not isinstance(c,bool)and c>0:hosts.setdefault(c,[]).append(e)
 keys=('install','execution','proof','restore','outside','uncorrelated','ambiguous')
 categories={c:{k:dict(events=0,duration_us=0.)for k in keys}for c in('kernel','gpu_memcpy','gpu_memset')};examples=[]
 for e in events:
  cat=e.get('cat')
  if e.get('ph')!='X'or cat not in categories:continue
  interval(e);c=e.get('args',{}).get('correlation');h=hosts.get(c,[])
  phases={phase(x)for x in h};p=next(iter(phases))if len(phases)==1 else'ambiguous'if phases else'uncorrelated'
  x=categories[cat][p];x['events']+=1;x['duration_us']+=e['dur']
  if p in('outside','uncorrelated','ambiguous')and len(examples)<16:examples.append(dict(category=cat,name=e.get('name'),correlation=c,classification=p))
 kernels=categories['kernel'];unknown=sum(kernels[k]['events']for k in('outside','uncorrelated','ambiguous'))
 ready=kernels['execution']['events']>0 and unknown==0
 return dict(complete=True,model_denominator_ready=ready,scope_names=names,phase_ranges={k:{f:e[f]for f in('pid','tid','ts','dur')}for k,e in ranges.items()},partition=categories,request_kernel_count=kernels['execution']['events'],request_kernel_duration_us=kernels['execution']['duration_us'],excluded_phase_kernel_count=sum(kernels[k]['events']for k in('install','proof','restore')),unresolved_kernel_count=unknown,unresolved_examples=examples,denominator='Sum of GPU kernel durations whose correlated CPU runtime/driver launch records lie wholly on the request_execution CPU range thread; not entire profiler session or GPU timestamp window',limits='API validators within execution are included in model request denominator; actual cuTile numerator requires separate verified attribution. Duration sum is not overlap-free busy time or wall time. Cross-thread/unresolved kernel launches block denominator readiness; transfers partitioned separately.')
