"""Two frozen slot-mapping cases, source-derived device allocations, no performance."""
import argparse,fcntl,json,os,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
MANIFEST=P/'slot_mapping_runtime_capture_v1/slotMappingCapture01/bank/MANIFEST.json'
MANIFEST_SHA='972e055a02c3075e638e5d74d3909cec0b8b2b4a060b19b6fc8c9c72767e734f'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-gpu',action='store_true');ap.add_argument('--output',required=True);a=ap.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(MANIFEST)==MANIFEST_SHA
 if not a.run_gpu:print(json.dumps(dict(executed=False,manifest_sha=MANIFEST_SHA,cases=2,placement='source-derived device allocation',timing=False)));return 0
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
  from tileir_vllm_slot_mapping_v1.api import compute_slot_mappings,reference_cpu
  def raw(t):return torch.empty(0,dtype=torch.uint8,device=t.device).set_(t.untyped_storage(),0,(t.untyped_storage().nbytes(),),(1,)).cpu().clone()
  def load(rec):
   assert sha(rec['storage_path'])==rec['storage_sha256'];v=torch.load(rec['storage_path'],map_location='cpu',weights_only=True)
   assert v.dtype==torch.uint8 and v.numel()==rec['metadata']['storage_bytes'];return v
  def restore(rec,device):
   m=rec['metadata'];b=load(rec).to(device);dtype=getattr(torch,m['dtype'].removeprefix('torch.'))
   return b.view(dtype).as_strided(m['shape'],m['stride'],m['storage_offset'])
  for sample in json.loads(MANIFEST.read_text())['samples']:
   pre=sample['pre'];post=sample['post'];padded=sample['batch']['num_tokens_after_padding'];phase=sample['phase'];assert sample['cp_size']==1
   names=['src_'+str(g) for g in range(4)]+['indices','starts','positions','sizes','enabled']
   cpu={k:restore(pre[k],'cpu') for k in names};expected=restore(pre['out'],'cpu')
   reference_cpu(tuple(cpu['src_'+str(g)] for g in range(4)),*(cpu[k] for k in names[4:]),expected,padded)
   assert torch.equal(raw(expected),load(post['out']))
   assert sample['python_block_sizes']==cpu['sizes'].tolist()==[128,128,128,528] and bool(cpu['enabled'].all())
   gpu={k:restore(pre[k],'cuda') for k in names};tables=tuple(gpu['src_'+str(g)] for g in range(4));actual_tokens=int(cpu['starts'][-1])
   row=dict(phase=phase,placement='device allocations matching source-derived creation path; not pointer identity',sizes=cpu['sizes'].tolist(),enabled=cpu['enabled'].tolist(),actual_tokens=actual_tokens,padded_tokens=padded,capacity=expected.shape[1],repeats=[])
   r['records'].append(row);save();prior=None
   for repeat in range(2):
    output=restore(pre['out'],'cuda');r.update(stage='target_launch_intent',target_phase=phase,target_repeat=repeat);save()
    view=compute_slot_mappings(tables,*(gpu[k] for k in names[4:]),output,padded,cp_size=1)
    torch.cuda.synchronize();r['gpu_executed']=True;actual=raw(output)
    exact=torch.equal(actual,load(post['out']));unchanged=all(torch.equal(raw(t),load(pre[k])) for k,t in gpu.items())
    md=sample['output']['metadata'];viewok=(view.untyped_storage().data_ptr()==output.untyped_storage().data_ptr() and list(view.shape)==md['shape'] and list(view.stride())==md['stride'] and view.storage_offset()==md['storage_offset'] and str(view.dtype)==md['dtype'])
    padok=bool((output[:,actual_tokens:]==-1).all().item());f=out/(phase+'_'+str(repeat)+'.pt');torch.save(actual,f)
    row['repeats'].append(dict(full_storage_raw_equal=exact,input_unchanged=unchanged,views_alias_expected=viewok,full_capacity_pad=padok,repeat_raw_equal=True if prior is None else torch.equal(prior,actual),path=str(f),sha256=sha(f)));save();prior=actual;del view,output
   del gpu,tables
  r.update(complete=True,stage='complete');verify_pins(R/'CPU-PINS.json');save()
  from slot_mapping_qualification_v1.result_gate import result_gate
  gate=result_gate(r);r['gate']=gate;save();return 0 if gate['raw_pass'] else 2
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
