"""Offline public CPU launch correlation; no GPU or source-role inference."""
import json,bisect
from collections import defaultdict,Counter
from pathlib import Path
def analyze(events):
 hosts=defaultdict(list);ranges=defaultdict(list)
 request=[e for e in events if e.get('cat')=='user_annotation' and e.get('name')=='metadata_request_execution::lean']
 assert len(request)==1;request=request[0]
 for e in events:
  if e.get('cat')in ('cuda_runtime','cuda_driver') and isinstance(e.get('args',{}).get('correlation'),int):hosts[e['args']['correlation']].append(e)
  if e.get('cat')=='user_annotation'and e.get('name','').startswith(('metadata_site::','python_helper_site::')):ranges[(e['pid'],e['tid'])].append(e)
 starts={};maxends={}
 for k,v in ranges.items():
  v.sort(key=lambda e:e['ts']);starts[k]=[e['ts']for e in v];mx=0;maxends[k]=[]
  for r in v:mx=max(mx,r['ts']+r['dur']);maxends[k].append(mx)
 counts=Counter();unknown=Counter();excluded=0;total=0
 for e in events:
  if e.get('cat')not in ('kernel','gpu_memcpy','gpu_memset'):continue
  hs=hosts.get(e.get('args',{}).get('correlation'),[])
  scopes=[];inside=[]
  for h in hs:
   inside.append(h['pid']==request['pid'] and h['tid']==request['tid'] and request['ts']<=h['ts'] and h['ts']+h['dur']<=request['ts']+request['dur'])
   key=(h['pid'],h['tid']);rs=ranges.get(key,[]);j=bisect.bisect_right(starts.get(key,[]),h['ts'])-1
   target_end=h['ts']+h['dur'];found=None
   # Same-thread record_function intervals nest; latest containing start is innermost.
   while j>=0 and maxends[key][j]>=target_end:
    r=rs[j]
    if r['ts']+r['dur']>=target_end:found=r['name'];break
    j-=1
   scopes.append(found)
  if inside and not any(inside):excluded+=1;continue
  total+=1
  if not hs or not all(inside)or len(set(scopes))!=1 or scopes[0]is None:unknown[(e['cat'],e['name'])]+=1
  else:counts[(e['cat'],scopes[0],e['name'])]+=1
 return dict(request_correlated_or_unresolved_events=total,excluded_events=excluded,attributed=[dict(category=k[0],site=k[1],symbol=k[2],count=n)for k,n in counts.items()],unknown=[dict(category=k[0],symbol=k[1],count=n)for k,n in unknown.items()],limits='Launch correlation plus same-thread annotation only; uncovered async launches remain unknown. Categories separate, no duration/coverage percentage claim.')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('output',type=Path);a=p.parse_args();assert not a.output.exists();a.output.write_text(json.dumps(analyze(json.loads(a.trace.read_text())['traceEvents']),indent=2)+'\n')
