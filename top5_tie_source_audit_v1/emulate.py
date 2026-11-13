from pathlib import Path
import torch,json,hashlib
R=Path(__file__).resolve().parent;P=R.parent
def network(values,ids):
 keys=list(values)+[0.]*(32-len(values));ids=list(ids)+[0]*(32-len(ids));valid=[True]*len(values)+[False]*(32-len(values))
 def step(stride,size=None):
  for tid in range(16):
   pos=2*tid-(tid&(stride-1));b=pos+stride;direction=False if size is None else bool(tid&(size//2))
   swap=(keys[pos]>keys[b] and valid[pos]) or not valid[b]
   if swap==direction:
    for a in (keys,ids,valid):a[pos],a[b]=a[b],a[pos]
 for size in (2,4,8,16):
  stride=size//2
  while stride:step(stride,size);stride//=2
 for stride in (16,8,4,2,1):step(stride)
 return ids[:5]
def emulate(x):
 result=[]
 for row in x.float():
  kth=float(torch.topk(row,5).values[-1]);above=torch.nonzero(row>kth).flatten().tolist();eq=torch.nonzero(row==kth).flatten().tolist();ids=above+eq[:5-len(above)];result.append(network([float(row[i])for i in ids],ids))
 return torch.tensor(result)
d=torch.load(P/'selected_ranks_runtime_integration_v3/samplerThreeIntegration01/shadow/first_mismatch/values.pt',map_location='cpu',weights_only=True);x=d['inputs']['logits'].view(torch.bfloat16).reshape(32,248320)
y=emulate(x);native=d['native_or_actual_ids'][:,1:];candidate=d['candidate_ids'][:,1:]
report=dict(scope='CPU public-source algorithm emulator, not observed CUDA dispatch',native_mismatches=int((y!=native).sum()),ascending_candidate_mismatches=int((candidate!=native).sum()),rows=[])
for j in range(32):
 if not torch.equal(candidate[j],native[j]):report['rows'].append(dict(row=j,native=native[j].tolist(),ascending=candidate[j].tolist(),emulated=y[j].tolist(),values=x[j,native[j]].float().tolist()))
m=json.loads((P/'greedy_sampler_runtime_capture_v1/greedySamplerCapture01/bank/MANIFEST.json').read_text());report['original_banks']=[]
for s in m['samples']:
 if s['kind']=='topk':
  a=torch.load(s['inputs']['logits']['path'],map_location='cpu',weights_only=True);b=torch.load(s['outputs']['logprob_token_ids']['path'],map_location='cpu',weights_only=True)[:,1:];report['original_banks'].append(dict(call=s['call_index'],mismatches=int((emulate(a)!=b).sum())))
assert not torch.cuda.is_initialized();report['cuda_initialized']=False
(R/'CPU-EMULATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
