"""Actual first/last greedy banks: public native2 then TileIR2, not top5 qualification."""
import argparse,json,os,sys,fcntl,signal,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
MANIFEST_SHA='8aff9ec850b5df8dcb1c08621e1328cf5d7fbfcddbaa78323616e0217d9d6b72'
def check_manifest(m):
 assert m['complete'] and m['observers_removed'] and len(m['rows'])==16
 cases=[s for s in m['samples'] if s['kind']=='gumbel'];assert [s['call_index'] for s in cases]==[0,15]
 for s in cases:
  assert s['observations']['all_active_temperature_zero']
  assert s['observations']['scalars']==dict(apply_temperature=False,is_drafting=False,logits_cache=None,logits_cache_col=None,use_fp64=False,logits_cache_source=None)
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
 r=dict(complete=False,gpu_executed=False,records=[],manifest_sha=a.manifest_sha,timing=False,scope='Two actual greedy banks only; no RNG/top5/logprob/rank/fullmodel support claim')
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.alarm(285)
 try:
  import torch
  from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
  from norm_runtime_contract_capture_v1.observer import identity,aliases
  from tileir_vllm_greedy_sample_v1 import greedy_sample
  from tp1_embedding_runtime_capture_v2.observer import meta,digest
  assert identity(gumbel_sample)==m['source']['gumbel'];torch.set_num_threads(4)
  flags=(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k);assert flags==(True,True);r['precision_flags_readonly']=list(flags)
  def load(item):
   x=torch.load(item['path'],weights_only=True,map_location='cpu');assert digest(x)==item['logical_sha256'];return x
  def restore(item):
   x=load(item);d=item['metadata'];assert list(x.shape)==d['shape'] and str(x.dtype)==d['dtype']
   size=d['storage_nbytes']//x.element_size();assert size*x.element_size()<=64*1024*1024
   base=torch.zeros(size,dtype=x.dtype,device='cuda');v=torch.as_strided(base,d['shape'],d['stride'],d['storage_offset']);v.copy_(x);return base,v
  for s in cases:
   expected=load(s['outputs']['sampled']);previous={}
   for arm in ('native','tile'):
    for repeat in (0,1):
     pairs={k:restore(item) for k,item in s['inputs'].items()};values={k:v for k,(b,v) in pairs.items()};before={k:digest(b) for k,(b,v) in pairs.items()}
     idx=values['expanded_idx_mapping'];assert bool((values['temperature'].index_select(0,idx)==0).all())
     r['gpu_executed']=True;save()
     with torch.inference_mode():
      out=gumbel_sample(**values,**s['observations']['scalars']) if arm=='native' else greedy_sample(values['logits'])
     torch.cuda.synchronize();actual=out.cpu();d=meta(out);od=s['outputs']['sampled']['metadata']
     raw=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8));layout=all(d[k]==od[k] for k in ('shape','stride','storage_offset','dtype'))
     preserved=all(digest(b)==before[k] for k,(b,v) in pairs.items());fresh=all(not x['storage_overlap'] for x in aliases({**values,'output':out}).values() if x.get('storage_overlap') is not None)
     # Actual captured inputs are pairwise disjoint. Reject unexpected replay overlap.
     rep=repeat==0 or torch.equal(actual,previous[arm]);valid=out.dtype==torch.int64 and bool(((out>=0)&(out<248320)).all())
     passed=raw and layout and preserved and fresh and rep and valid
     row=dict(call_index=s['call_index'],phase=s['phase'],arm=arm,repeat=repeat,raw_equal=raw,integer_exact=raw,strict_failures=int((actual!=expected).sum()),output_valid_integer_range=valid,layout_equal=layout,input_full_owned_storage_preserved=preserved,fresh_disjoint=fresh,repeat_exact=rep,passed=passed);r['records'].append(row);save()
     if not passed:torch.save(actual,a.output/(str(s['call_index'])+'_'+arm+str(repeat)+'_failed.pt'));r['raw_difference_hold']=True;save();return 2
     previous[arm]=actual;del pairs,values,out
  assert flags==(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
  verify_pins(R/'CPU-PINS.json');r['complete']=True;r['gate']=result_gate(r);save();return 0 if r['gate']['raw_pass'] else 1
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
