from pathlib import Path
import argparse,json,os,sys,signal,traceback
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
class Hold(Exception):pass
def result_gate(r):return dict(raw_pass=r.get('complete')is True and len(r.get('records',[]))==12 and all(x.get('passed')is True for x in r['records']),raw_difference_hold=r.get('hold')is True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args();assert sha(a.manifest)==a.manifest_sha;verify_pins(R/'CPU-PINS.json');fixtures=json.loads(Path(a.manifest).read_text())['fixtures']
 if not a.run_gpu:print('CPU pins/3 synthetic fixtures PASS; GPU0');return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);s=os.fstat(fd);t=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').stat();assert(s.st_dev,s.st_ino)==(t.st_dev,t.st_ino)
 a.output.mkdir(exist_ok=False)
 for env,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/name).mkdir();os.environ[env]=str(a.output/name)
 r=dict(complete=False,manifest_sha=a.manifest_sha,records=[],synthetic_only=True,actual_owner_values=False,timing=False)
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('600s synthetic bound')));signal.alarm(600);save()
 try:
  import torch
  from tileir_gdn_parent_state_v2 import gather_state,mask_initial_state_,scatter_state_
  def same(x,y):
   assert x.shape==y.shape and x.dtype==y.dtype
   # Complete raw-byte comparison, bounded8MiB chunks, not a probabilistic hash.
   xx=x.contiguous().view(torch.uint8).flatten();yy=y.contiguous().view(torch.uint8).flatten()
   for n in range(0,xx.numel(),8*1024**2):
    if not torch.equal(xx[n:n+8*1024**2].cpu(),yy[n:n+8*1024**2].cpu()):return False
   return True
  def inject_bits(t):
   # Private synthetic tensors only. Integer writes preserve exact NaN payloads.
   pattern=torch.tensor([0x00000000,0x80000000,0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x7f812345],dtype=torch.uint32,device='cpu').to('cuda')
   t.view(torch.uint32).flatten()[:7]=pattern
  def seed(f):
   n,h,k,v=f['shape'];inner=h*k*v;length=f['offset']+(n-1)*f['pitch']+inner+97;assert length*4<=384*1024**2
   backing=torch.empty(length,device='cuda',dtype=torch.float32)
   bits=backing.view(torch.int32)
   for lo in range(0,length,1024*1024):
    ix=torch.arange(lo,min(length,lo+1024*1024),device='cuda',dtype=torch.int64);values=((ix%1021)+0x3f000000).to(torch.int32);values=torch.where(ix%7==0,-2147483648,values);values=torch.where(ix%11==0,0,values);bits[lo:lo+len(ix)]=values
   inject_bits(backing)
   owner=backing.as_strided(tuple(f['shape']),(f['pitch'],k*v,v,1),f['offset'])
   for row in f['indices'][::f['index_stride']]:inject_bits(owner[row])
   return backing,owner
  with torch.inference_mode():
   for f in fixtures:
    ref=None
    for arm in ('native','tile'):
     for rep in (0,1):
      row=dict(fixture=f['name'],arm=arm,repeat=rep,passed=False);r['records'].append(row);save()
      backing,owner=seed(f);indexbuf=torch.tensor(f['indices'],device='cuda',dtype=torch.int64 if f['index64']else torch.int32);indices=indexbuf[::f['index_stride']];mask=torch.tensor(f['mask'],device='cuda',dtype=torch.bool)
      input_indices=indices.cpu();input_mask=mask.cpu();before=backing.clone() # Syntheticprivatebounded backing ONLY, not live sharedowner.
      gathered=owner[indices] if arm=='native'else gather_state(owner,indices)
      assert gathered.is_contiguous() and gathered.storage_offset()==0 and gathered.untyped_storage().data_ptr()!=owner.untyped_storage().data_ptr()
      assert same(backing,before);gcopy=gathered.clone()
      if arm=='native':gathered[~mask,...]=0
      else:assert mask_initial_state_(gathered,mask)is gathered
      assert same(backing,before)
      last=torch.full_like(gathered,1.25);last.view(torch.int32).flatten()[::5]=-2147483648
      for ii in range(last.shape[0]):inject_bits(last[ii])
      lastcopy=last.clone()
      if arm=='native':owner[indices]=last
      else:assert scatter_state_(owner,indices,last)is owner
      assert same(last,lastcopy) and torch.equal(indices.cpu(),input_indices) and torch.equal(mask.cpu(),input_mask)
      # Bitcopy API intentionally supports Inf/NaN payloads; raw comparison is the gate.
      for ii,keep in enumerate(f['mask']):
       if keep:assert same(gathered[ii],gcopy[ii])
       else:assert bool((gathered[ii].view(torch.int32)==0).all())
      row['mask_true_bits_preserved_false_positive_zero']=True
      if ref is None:ref=(backing,gcopy,gathered);row['reference']=True
      else:
       checks=dict(entire_backing_raw=same(backing,ref[0]),gather_raw=same(gcopy,ref[1]),masked_raw=same(gathered,ref[2]));row.update(checks)
       if not all(checks.values()):r['hold']=True;save();raise Hold(f['name'])
      # Includesprefix,suffix,padding andunselectedrows: fullbackingbyte oracle.
      row['passed']=True;save();del before,last,lastcopy,owner,indexbuf,indices,mask
      if not row.get('reference'):del backing,gcopy,gathered
    del ref
  r['complete']=True;verify_pins(R/'CPU-PINS.json');save();return 0
 except Hold:r['hold']=True;save();return 2
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0);save()
if __name__=='__main__':raise SystemExit(main())
