"""Actual first/last selected-logprob banks: public function2 then TileIR2, not topk selection."""
import argparse,json,os,sys,fcntl,signal,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
MANIFEST_SHA='8aff9ec850b5df8dcb1c08621e1328cf5d7fbfcddbaa78323616e0217d9d6b72'
def check_manifest(m):
 assert m['complete'] and m['observers_removed'] and len(m['rows'])==16
 cases=[s for s in m['samples'] if s['kind']=='topk'];assert [s['call_index'] for s in cases]==[0,15]
 for s in cases:
  assert s['inputs']['logits']['metadata']['dtype']=='torch.bfloat16'
  assert s['observations']['scalars']==dict(num_logprobs=5,cu_num_logits=None,max_per_req_token_ids=0,logits_mode=False)
  assert all(s['observations']['input_logical_preserved'].values())
 return cases
def result_gate(r):
 rows=r.get('records',[]);expected=[(c,a,n) for c in (0,15) for a in ('native','tile') for n in (0,1)]
 passed=r.get('complete') is True and [(x['call_index'],x['arm'],x['repeat']) for x in rows]==expected and all(x['passed'] for x in rows)
 return dict(raw_pass=passed,contract_pass=passed,complete=r.get('complete') is True,raw_difference_hold=r.get('raw_difference_hold',False))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 verify_pins(R/'CPU-PINS.json');assert a.manifest_sha==MANIFEST_SHA==sha(a.manifest);m=json.loads(a.manifest.read_text());cases=check_manifest(m)
 for item in m['source'].values():assert sha(item['source'])==item['source_sha256']
 for s in cases:
  for item in [*s['inputs'].values(),*s['outputs'].values()]:assert sha(item['path'])==item['sha256']
 if not a.run_gpu:print(json.dumps(dict(executed=False,cases=2,ordered_calls=8)));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);lock='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock';assert os.fstat(fd).st_ino==os.stat(lock).st_ino;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for k,n in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TRITON_CACHE_DIR','triton'),('TMPDIR','tmp')]:
  (a.output/n).mkdir();os.environ[k]=str(a.output/n)
 r=dict(complete=False,gpu_executed=False,records=[],manifest_sha=a.manifest_sha,timing=False,scope='Two actual selected-logprob banks only; no topk selection/ranks/fullmodel claim')
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.alarm(285)
 try:
  import torch
  from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs as native
  import inspect
  from norm_runtime_contract_capture_v1.observer import identity,aliases
  from tileir_vllm_selected_logprobs_v1 import compute_token_logprobs as tile
  from gdn_post_conv_qualification_v2.run import compare
  from tp1_embedding_runtime_capture_v2.observer import meta,digest
  assert sha(inspect.getsourcefile(native))==m['source']['topk']['source_sha256'];torch.set_num_threads(4)
  flags=(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k);assert flags==(True,True);r['precision_flags_readonly']=list(flags)
  def load(item):
   x=torch.load(item['path'],weights_only=True,map_location='cpu');assert digest(x)==item['logical_sha256'];return x
  def restore(item):
   x=load(item);d=item['metadata'];assert list(x.shape)==d['shape'] and str(x.dtype)==d['dtype']
   size=d['storage_nbytes']//x.element_size();assert size*x.element_size()<=64*1024*1024
   base=torch.zeros(size,dtype=x.dtype,device='cuda');v=torch.as_strided(base,d['shape'],d['stride'],d['storage_offset']);v.copy_(x);return base,v
  for s in cases:
   expected=load(s['outputs']['logprobs']);previous={}
   for arm in ('native','tile'):
    for repeat in (0,1):
     pairs={'logits':restore(s['inputs']['logits']),'token_ids':restore(s['outputs']['logprob_token_ids'])};values={k:v for k,(b,v) in pairs.items()};before={k:digest(b) for k,(b,v) in pairs.items()}
     x=values['logits'];ids=values['token_ids'];assert bool(((ids>=0)&(ids<248320)).all())
     assert bool(torch.isfinite(x.gather(1,ids)).all()) and not bool(torch.isnan(x).any()) and not bool(torch.isposinf(x).any())
     r['gpu_executed']=True;save()
     with torch.inference_mode():
      if arm=='native':
       out=native(x,ids)
      else:out=tile(x,ids)
     torch.cuda.synchronize();actual=out.cpu();d=meta(out);od=s['outputs']['logprobs']['metadata']
     raw=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8));layout=all(d[k]==od[k] for k in ('shape','stride','storage_offset','dtype'))
     preserved=all(digest(b)==before[k] for k,(b,v) in pairs.items());fresh=all(not x['storage_overlap'] for x in aliases({**values,'output':out}).values() if x.get('storage_overlap') is not None)
     # Actual captured inputs are pairwise disjoint. Reject unexpected replay overlap.
     rep=repeat==0 or torch.equal(actual,previous[arm]);valid=out.dtype==torch.float32 and bool(torch.isfinite(out).all());comparison=compare(expected,actual,torch)
     passed=raw and layout and preserved and fresh and rep and valid and comparison['strict_failures']==0
     row=dict(call_index=s['call_index'],phase=s['phase'],arm=arm,repeat=repeat,raw_equal=raw,comparison=comparison,strict_failures=comparison['strict_failures'],output_finite=valid,layout_equal=layout,input_full_owned_storage_preserved=preserved,fresh_disjoint=fresh,repeat_exact=rep,passed=passed);r['records'].append(row);save()
     if not passed:torch.save(actual,a.output/(str(s['call_index'])+'_'+arm+str(repeat)+'_failed.pt'));r['raw_difference_hold']=True;save();return 2
     previous[arm]=actual;del pairs,values,out
  assert flags==(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
  verify_pins(R/'CPU-PINS.json');r['complete']=True;r['gate']=result_gate(r);save();return 0 if r['gate']['raw_pass'] else 1
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
