"""Two actual sample-count cases, native/TileIR repeats and mapped-UVA preservation."""
import argparse,fcntl,json,os,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
MANIFEST=P/'sample_counts_runtime_capture_v1/sampleCountsCapture01/bank/MANIFEST.json'
MANIFEST_SHA='72cadfe5f8e8d78c0ac6334dc2ecac43e09986c16f8e945389d80ef35ba825ea'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-gpu',action='store_true');ap.add_argument('--output',required=True);a=ap.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(MANIFEST)==MANIFEST_SHA
 if not a.run_gpu:print(json.dumps(dict(executed=False,manifest_sha=MANIFEST_SHA,cases=2,placement='actual mapped UVA',arms=['native','tileir'],repeats=2,timing=False)));return 0
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
  from tileir_vllm_sample_counts_v1.api import get_num_sampled_and_rejected as tile,reference_cpu
  from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected as native
  from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
  def raw(t):return torch.empty(0,dtype=torch.uint8,device=t.device).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,)).cpu().clone()
  def load(rec):
   assert sha(rec['storage_path'])==rec['storage_sha256'];v=torch.load(rec['storage_path'],map_location='cpu',weights_only=True)
   assert v.dtype==torch.uint8 and v.numel()==rec['metadata']['storage_bytes'];return v
  def restore(rec,device):
   m=rec['metadata'];b=load(rec).to(device);dtype=getattr(torch,m['dtype'].removeprefix('torch.'))
   return b.view(dtype).as_strided(m['shape'],m['stride'],m['storage_offset'])
  for sample in json.loads(MANIFEST.read_text())['samples']:
   pre=sample['pre'];phase=sample['phase'];names=['sampled','seq_lens','cu_logits','indices','prefill']
   cpu=[restore(pre[k],'cpu') for k in names];cpuout=reference_cpu(*cpu)
   assert all(torch.equal(raw(t),load(rec)) for t,rec in zip(cpuout,sample['outputs']))
   place=sample['prefill_placement'];assert place['buffer_class']=='UvaBuffer' and place['cpu']['is_pinned'] and place['same_active_backing']
   hostbytes=torch.empty(load(pre['prefill']).shape,dtype=torch.uint8,pin_memory=True);hostbytes.copy_(load(pre['prefill']))
   md=pre['prefill']['metadata'];owner=hostbytes.view(torch.int32).as_strided(md['shape'],md['stride'],md['storage_offset']);prefill=get_accelerator_view_from_cpu_tensor(owner)
   assert owner.is_pinned() and prefill.device.type=='cuda' and prefill.data_ptr()==owner.data_ptr()
   readonly=[restore(pre[k],'cuda') for k in names[1:4]]+[prefill]
   for arm,fn in [('native',native),('tileir',tile)]:
    row=dict(phase=phase,arm=arm,placement='actual mapped-UVA primary',mapped_owner_pinned=owner.is_pinned(),mapped_pointer_equal=owner.data_ptr()==prefill.data_ptr(),repeats=[]);r['records'].append(row);save();prior=None
    for repeat in range(2):
     sampled=restore(pre['sampled'],'cuda');r.update(stage='target_call_intent',target_phase=phase,target_arm=arm,target_repeat=repeat);save()
     # Exactly one original public/native or candidate invocation per repetition.
     values=fn(sampled,*readonly);torch.cuda.synchronize();r['gpu_executed']=True
     assert isinstance(values,tuple) and len(values)==2
     actual=[raw(t) for t in values];equals=[torch.equal(t,load(rec)) for t,rec in zip(actual,sample['outputs'])]
     unchanged=all(torch.equal(raw(t),load(pre[k])) for k,t in zip(names[1:],readonly)) and torch.equal(raw(owner),load(pre['prefill']))
     def overlaps(a,b):
      x,y=a.untyped_storage(),b.untyped_storage();return x.data_ptr()<y.data_ptr()+y.nbytes() and y.data_ptr()<x.data_ptr()+x.nbytes()
     identity=values[0] is sampled;disjoint=all(not overlaps(values[1],t) for t in [sampled,*readonly])
     layouts=all(list(t.shape)==rec['metadata']['shape'] and list(t.stride())==rec['metadata']['stride'] and t.storage_offset()==rec['metadata']['storage_offset'] and str(t.dtype)==rec['metadata']['dtype'] for t,rec in zip(values,sample['outputs']))
     f=out/(phase+'_'+arm+'_'+str(repeat)+'.pt');torch.save(actual,f)
     rep=dict(full_storage_raw_equal=equals,input_unchanged=unchanged,first_return_same_object=identity,rejected_disjoint=disjoint,output_layout_equal=layouts,repeat_raw_equal=True if prior is None else all(torch.equal(x,y) for x,y in zip(prior,actual)),call_count=1,path=str(f),sha256=sha(f));row['repeats'].append(rep);save()
     if not (all(equals) and unchanged and identity and disjoint and layouts and rep['repeat_raw_equal']):
      r.update(raw_difference_hold=True,stage='first_difference_stop');save();return 2
     prior=actual;del values,sampled
   del readonly,prefill,owner,hostbytes
  r.update(complete=True,stage='complete');verify_pins(R/'CPU-PINS.json');save()
  from sample_counts_qualification_v1.result_gate import result_gate
  gate=result_gate(r);r['gate']=gate;save();return 0 if gate['raw_pass'] else 2
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
