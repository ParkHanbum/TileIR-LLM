"""Saved call8 replay only; no model, no raw criterion relaxation."""
import argparse,json,os,sys,fcntl,signal,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins

def result_gate(r):
 rows=r.get('records',[])
 ordered=[(x['arm'],x['repeat']) for x in rows]==[(a,n) for a in ('native','tile') for n in (0,1)]
 complete=r.get('complete') is True and ordered and all(x['binding_pass'] for x in rows)
 return dict(complete=complete,contract_pass=complete,raw_pass=complete and all(x['vs_native']['raw_equal'] for x in rows),raw_difference_hold=complete and any(not x['vs_native']['raw_equal'] for x in rows))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha
 m=json.loads(a.manifest.read_text());assert sha(Path(m['bank']))==m['bank_sha'];assert sha(Path(m['progress']))==m['progress_sha'];assert m['row']['batch']['num_computed_tokens']==[59]*32
 if not a.run_gpu:print(json.dumps(dict(executed=False,calls=4,original_topk_ids_missing=True)));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);lock='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock';assert os.fstat(fd).st_ino==os.stat(lock).st_ino;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for k,n in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TRITON_CACHE_DIR','triton'),('TMPDIR','tmp')]:
  (a.output/n).mkdir();os.environ[k]=str(a.output/n)
 r=dict(complete=False,manifest_sha=a.manifest_sha,records=[],gpu_executed=False,original_topk_ids_missing=True,timing=False)
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.alarm(285)
 try:
  import torch
  from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs as native
  from tileir_vllm_selected_logprobs_v1 import compute_token_logprobs as tile
  from gdn_post_conv_qualification_v2.run import compare
  torch.set_num_threads(4)
  flags=lambda:(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
  assert flags()==(True,True);r['native_flags']=[True,True]
  bank=torch.load(m['bank'],map_location='cpu',weights_only=True);expected=bank['native_or_actual_scores'];savedtile=bank['candidate_scores'];previous={};priorids=None
  for arm in ('native','tile'):
   for repeat in (0,1):
    owners={};vals={}
    for name,raw in bank['inputs'].items():
     d=m['row']['inputs'][name];assert raw.dtype==torch.uint8 and raw.numel()==d['storage_nbytes']
     owner=raw.clone().cuda();dt=getattr(torch,d['dtype'].split('.')[-1]);view=torch.as_strided(owner.view(dt),d['shape'],d['stride'],d['storage_offset']);owners[name]=owner;vals[name]=view
    x=vals['logits'];chosen=vals['sampled_token_ids'];r['gpu_executed']=True;save()
    with torch.inference_mode():
     ids=torch.cat((chosen.unsqueeze(-1),torch.topk(x,5,dim=-1).indices),dim=1)
     ids_before=ids.clone();out=(native if arm=='native' else tile)(x,ids)
    torch.cuda.synchronize();actual=out.cpu();actualids=ids.cpu();torch.save(actual,a.output/f'{arm}{repeat}-scores.pt');torch.save(actualids,a.output/f'{arm}{repeat}-ids.pt')
    preserved=all(torch.equal(v.cpu(),bank['inputs'][k]) for k,v in owners.items()) and torch.equal(ids,ids_before)
    layout=out.dtype==torch.float32 and tuple(out.shape)==(32,6) and out.stride()==(6,1) and out.storage_offset()==0
    fresh=out.untyped_storage().data_ptr() not in [v.untyped_storage().data_ptr() for v in [*owners.values(),ids]]
    vs=compare(expected,actual,torch);own=compare(expected if arm=='native' else savedtile,actual,torch)
    rep=repeat==0 or torch.equal(actual.view(torch.uint8),previous[arm].view(torch.uint8));idrep=priorids is None or torch.equal(actualids,priorids)
    binding=own['raw_equal'] and preserved and layout and fresh and rep and idrep and own['candidate_nonfinite']==0 and vs['strict_failures']==0
    r['records'].append(dict(arm=arm,repeat=repeat,vs_native=vs,vs_saved_same_arm=own,input_storage_preserved=preserved,layout=layout,fresh=fresh,repeat_exact=rep,regenerated_ids_repeat_exact=idrep,binding_pass=binding));save()
    if not binding:r['binding_hold']=True;save();return 2
    previous[arm]=actual;priorids=actualids
  assert flags()==(True,True);verify_pins(R/'CPU-PINS.json');r['complete']=True;r['gate']=result_gate(r);save();return 0 if r['gate']['raw_pass'] else 2
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
