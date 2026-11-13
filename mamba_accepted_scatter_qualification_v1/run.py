import argparse,json,os,fcntl,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
def order():return [(c,a,r)for c in [*['actual%02d'%i for i in range(1,17)],'synthetic_changed','synthetic_negative']for a in ('native','tile')for r in (0,1)]
def result_gate(r):
 rows=r.get('records',[]);keys=('raw_equal','inputs_preserved','layout','return_none','repeat_exact','passed')
 ok=r.get('complete')is True and [(x['case'],x['arm'],x['repeat'])for x in rows]==order() and all(all(x.get(k)is True for k in keys)for x in rows)
 return dict(raw_pass=ok,raw_difference_hold=r.get('raw_difference_hold',False))
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha;m=json.loads(a.manifest.read_text())
 for k in ('source','bank_manifest','capture_result','capture_recovery'):assert sha(m[k])==m[k+'_sha256']
 assert [c['name']for c in m['cases']]==[x[0]for x in order()[::4]]
 for c in m['cases']:
  if not c['synthetic']:assert sha(c['bank'])==c['sha256']
 if not a.run_gpu:print(json.dumps(dict(executed=False,calls=72,scope=m['scope'])));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.stat('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock');f=os.fstat(fd);assert(st.st_dev,st.st_ino)==(f.st_dev,f.st_ino);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for env,n in [('CUDA_TILE_CACHE_DIR','tile'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton')]:
  (a.output/n).mkdir();os.environ[env]=str(a.output/n)
 r=dict(complete=False,gpu_executed=False,records=[],manifest_sha=a.manifest_sha,scope=m['scope'],timing=False)
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 signal.alarm(285);save()
 try:
  import torch
  from tileir_vllm_mamba_accepted_scatter_v1.api import scatter_num_accepted_,validate_indices
  from vllm.v1.worker.gpu.model_states.mamba_hybrid import _scatter_num_accepted_kernel
  torch.set_num_threads(4);flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k];assert flags()==[True,True]
  for c in m['cases']:
   if not c['synthetic']:
    bank=torch.load(c['bank'],map_location='cpu',weights_only=False);row=bank['row'];desc=row['descriptors'];assert all(desc[k]['offset']==0 and desc[k]['stride']==[1] and desc[k]['shape']==[32]for k in desc)
    indices=bank['indices'];samples=bank['sampled'];before=bank['owner_before'];expected=bank['owner_after'];offset=0;capacity=32
    assert torch.equal(indices.view(torch.uint8),bank['indices_owner_before']) and torch.equal(samples.view(torch.uint8),bank['sampled_owner_before'])
   else:
    indices=torch.arange(32,dtype=torch.int64);samples=torch.tensor(([-2,0,1,7,2147483647]*7)[:32],dtype=torch.int32);offset=3;capacity=40
    if c['name']=='synthetic_negative':indices[:2]=torch.tensor([-1,-7],dtype=torch.int64)
    before=torch.full((48,),-99,dtype=torch.int32).view(torch.uint8);expected=before.clone()
    for i,v in zip(indices.tolist(),samples.tolist()):
     if i>=0:expected.view(torch.int32)[offset+i]=max(v,1)
   validate_indices(indices.tolist(),capacity);previous={}
   for arm in ('native','tile'):
    for repeat in (0,1):
     I=indices.to('cuda');S=samples.to('cuda');owner=before.to('cuda');D=owner.view(torch.int32)[offset:offset+capacity];ptr=D.untyped_storage().data_ptr();r['gpu_executed']=True;save()
     with torch.inference_mode():
      if arm=='native':
       _scatter_num_accepted_kernel[(32,)](I,S,D);out=None # surrounding public helper returns None; launcher return is not an API return
      else:out=scatter_num_accepted_(I,S,D)
     torch.cuda.synchronize();actual=owner.cpu();raw=torch.equal(actual,expected);inputs=torch.equal(I.cpu().view(torch.uint8),indices.view(torch.uint8)) and torch.equal(S.cpu().view(torch.uint8),samples.view(torch.uint8));layout=D.dtype==torch.int32 and D.shape==(capacity,) and D.stride()==(1,) and D.storage_offset()==offset and D.untyped_storage().data_ptr()==ptr and D.untyped_storage().nbytes()==before.numel();rep=repeat==0 or torch.equal(actual,previous[arm]);passed=raw and inputs and layout and out is None and rep
     r['records'].append(dict(case=c['name'],synthetic=c['synthetic'],arm=arm,repeat=repeat,raw_equal=raw,inputs_preserved=inputs,layout=layout,return_none=out is None,repeat_exact=rep,passed=passed));save()
     if not passed:torch.save(dict(indices=indices,samples=samples,before=before,actual=actual,expected=expected,case=c),a.output/'FIRST-FAILURE.pt');r['raw_difference_hold']=True;save();return 2
     previous[arm]=actual
  assert flags()==[True,True];verify_pins(R/'CPU-PINS.json');r['complete']=True;save();return 0
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
