"""Derived actual-value fixtures; native cat2 then TileIR2 per first/last, no model proof."""
import argparse,json,os,fcntl,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
def result_gate(r):
 rows=r.get('records',[]);order=[(i,a,n)for i in (0,15)for a in ('native','tile')for n in (0,1)]
 ok=r.get('complete') is True and [(x['call_index'],x['arm'],x['repeat'])for x in rows]==order and all(x['passed']for x in rows)
 return dict(raw_pass=ok,contract_pass=ok,raw_difference_hold=r.get('raw_difference_hold',False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha;m=json.loads(a.manifest.read_text());assert [s['call_index']for s in m['samples']]==[0,15]
 assert sha(m['original_manifest'])==m['original_manifest_sha256'];assert sha(m['actual_trace_contract']['path'])==m['actual_trace_contract']['sha256']
 for s in m['samples']:
  for k in ('chosen','ids'):assert sha(s[k]['path'])==s[k]['sha256']
 if not a.run_gpu:print(json.dumps(dict(executed=False,calls=8,scope=m['scope'])));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);lock='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock';st=os.stat(lock);f=os.fstat(fd);assert(st.st_dev,st.st_ino)==(f.st_dev,f.st_ino);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for env,n in [('CUDA_TILE_CACHE_DIR','tile'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/n).mkdir();os.environ[env]=str(a.output/n)
 r=dict(complete=False,gpu_executed=False,records=[],manifest_sha=a.manifest_sha,scope=m['scope'],timing=False)
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 signal.alarm(285);save()
 try:
  import torch
  from tileir_vllm_sampler_cat_v1 import concat_sampled_top5
  from tp1_embedding_runtime_capture_v2.observer import digest
  torch.set_num_threads(4);flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k];assert flags()==[True,True]
  for s in m['samples']:
   def load(k):
    t=torch.load(s[k]['path'],map_location='cpu',weights_only=True);assert digest(t)==s[k]['logical_sha256'];return t
   chosen=load('chosen');expected=load('ids');assert chosen.dtype==expected.dtype==torch.int64 and tuple(chosen.shape)==(32,) and tuple(expected.shape)==(32,6) and torch.equal(chosen,expected[:,0])
   # Materialize contiguous independent owners; top5 is derived from actual saved output.
   c=chosen.clone().reshape(32,1);t=expected[:,1:].contiguous();previous={}
   for arm in ('native','tile'):
    for repeat in (0,1):
     x,y=c.cuda(),t.cuda();before=(digest(x),digest(y));r['gpu_executed']=True;save()
     with torch.inference_mode():out=torch.cat((x,y),dim=1)if arm=='native'else concat_sampled_top5(x,y)
     torch.cuda.synchronize();actual=out.cpu();raw=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8));layout=out.dtype==torch.int64 and tuple(out.shape)==(32,6)and out.stride()==(6,1)and out.storage_offset()==0 and out.untyped_storage().nbytes()==1536
     preserved=before==(digest(x),digest(y));fresh=out.untyped_storage().data_ptr()not in(x.untyped_storage().data_ptr(),y.untyped_storage().data_ptr());rep=repeat==0 or torch.equal(actual,previous[arm]);passed=raw and layout and preserved and fresh and rep
     row=dict(call_index=s['call_index'],arm=arm,repeat=repeat,raw_equal=raw,integer_exact=raw,strict_failures=int((actual!=expected).sum()),layout=layout,input_full_owned_bytes_preserved=preserved,fresh=fresh,repeat_exact=rep,passed=passed);r['records'].append(row);save()
     if not passed:torch.save(dict(chosen=x.cpu(),top5=y.cpu(),expected=expected,actual=actual),a.output/'FIRST-FAILURE.pt');r['raw_difference_hold']=True;save();return 2
     previous[arm]=actual
  assert flags()==[True,True];verify_pins(R/'CPU-PINS.json');r['complete']=True;save();return 0
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
