"""Two frozen cases, primary mapped-UVA counts and separate device-copy control."""
import argparse,fcntl,json,os,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
MANIFEST=P/'block_gather_runtime_capture_v1/blockGatherCapture01/bank/MANIFEST.json'
MANIFEST_SHA='e4daebf788a029eaa98f0200b8f482362772f5efbf8eded7fc3a93d315add912'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-gpu',action='store_true');ap.add_argument('--output',required=True);a=ap.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(MANIFEST)==MANIFEST_SHA
 if not a.run_gpu:print(json.dumps(dict(executed=False,manifest_sha=MANIFEST_SHA,cases=2,placements=['mapped_uva_primary','device_copy_control'],timing=False)));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);lock='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock'
 assert (os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(os.stat(lock).st_dev,os.stat(lock).st_ino);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=Path(a.output);out.mkdir(exist_ok=False)
 for key,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (out/name).mkdir();os.environ[key]=str(out/name)
 r=dict(complete=False,gpu_executed=False,manifest_sha=MANIFEST_SHA,records=[],timing=False,stage='started')
 def save():(out/'RESULT.json').write_text(json.dumps(r,indent=2))
 def timeout(*_):raise TimeoutError('285 second bounded child')
 signal.signal(signal.SIGALRM,timeout);signal.alarm(285);save()
 try:
  import torch
  from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
  from tileir_vllm_block_gather_v1.api import gather_block_tables,reference_cpu
  def raw(t):return torch.empty(0,dtype=torch.uint8,device=t.device).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,)).cpu().clone()
  def load(rec):
   assert sha(rec['storage_path'])==rec['storage_sha256'];v=torch.load(rec['storage_path'],map_location='cpu',weights_only=True)
   assert v.dtype==torch.uint8 and v.numel()==rec['metadata']['storage_bytes'];return v
  def restore(rec,device):
   m=rec['metadata'];b=load(rec).to(device);dtype=getattr(torch,m['dtype'].removeprefix('torch.'))
   return b.view(dtype).as_strided(m['shape'],m['stride'],m['storage_offset'])
  for sample in json.loads(MANIFEST.read_text())['samples']:
   pre=sample['pre'];post=sample['post'];padded=sample['batch']['num_reqs_after_padding'];phase=sample['phase']
   assert sample['counts_placement']['buffer_class']=='UvaBuffer'
   # Validate content bounds and native frozen mutation before any target launch.
   cs=tuple(restore(pre['src_'+str(g)],'cpu') for g in range(4));co=tuple(restore(pre['out_'+str(g)],'cpu') for g in range(4));cc=restore(pre['counts_device_view'],'cpu');ci=restore(pre['indices'],'cpu')
   reference_cpu(cs,cc,ci,co,padded)
   assert all(torch.equal(raw(co[g]),load(post['out_'+str(g)])) for g in range(4))
   for placement in ('mapped_uva_primary','device_copy_control'):
    src=tuple(restore(pre['src_'+str(g)],'cuda') for g in range(4));indices=restore(pre['indices'],'cuda')
    owner=None
    if placement=='mapped_uva_primary':
     # Public helper identical to UvaBuffer: mapped pinned CPU, no device copy.
     host_bytes=torch.empty(load(pre['counts_device_view']).shape,dtype=torch.uint8,pin_memory=True);host_bytes.copy_(load(pre['counts_device_view']))
     m=pre['counts_device_view']['metadata'];owner=host_bytes.view(torch.int32).as_strided(m['shape'],m['stride'],m['storage_offset'])
     counts=get_accelerator_view_from_cpu_tensor(owner)
     assert owner.is_pinned() and counts.device.type=='cuda' and counts.data_ptr()==owner.data_ptr()
    else:counts=restore(pre['counts_device_view'],'cuda')
    row=dict(phase=phase,placement=placement,counts_dtype=str(counts.dtype),indices_dtype=str(indices.dtype),counts_shape=list(counts.shape),mapped_owner_pinned=None if owner is None else owner.is_pinned(),mapped_pointer_equal=None if owner is None else owner.data_ptr()==counts.data_ptr(),repeats=[])
    prior=None
    for repeat in range(2):
     outputs=tuple(restore(pre['out_'+str(g)],'cuda') for g in range(4));r.update(stage='target_launch_intent',target_phase=phase,target_placement=placement,target_repeat=repeat);save()
     views=gather_block_tables(src,counts,indices,outputs,padded);torch.cuda.synchronize();r['gpu_executed']=True
     actual=[raw(t) for t in outputs];equals=[torch.equal(v,load(post['out_'+str(g)])) for g,v in enumerate(actual)]
     unchanged=all(torch.equal(raw(t),load(pre['src_'+str(g)])) for g,t in enumerate(src)) and torch.equal(raw(indices),load(pre['indices'])) and torch.equal(raw(counts),load(pre['counts_device_view']))
     if owner is not None:unchanged=unchanged and torch.equal(raw(owner),load(pre['counts_device_view']))
     aliases=all(v.untyped_storage().data_ptr()==o.untyped_storage().data_ptr() and tuple(v.shape)==(padded,o.shape[1]) and v.stride()==o.stride() for v,o in zip(views,outputs))
     path=out/(phase+'_'+placement+'_'+str(repeat)+'.pt');torch.save(actual,path)
     row['repeats'].append(dict(all_group_storage_raw_equal=equals,input_unchanged=unchanged,views_alias_expected=aliases,repeat_raw_equal=True if prior is None else all(torch.equal(x,y) for x,y in zip(prior,actual)),path=str(path),sha256=sha(path)))
     prior=actual;del views,outputs
    r['records'].append(row);save();del counts,owner,src,indices
  r.update(complete=True,stage='complete');verify_pins(R/'CPU-PINS.json');save()
  from block_gather_qualification_v1.result_gate import result_gate
  gate=result_gate(r);r['gate']=gate;save();return 0 if gate['raw_pass'] else 2
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
