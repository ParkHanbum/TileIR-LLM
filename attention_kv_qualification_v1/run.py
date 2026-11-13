"""Actual KV helper replay. Owned compact-prefix cache, no shared engine clone."""
import argparse,json,os,sys,signal,traceback,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins

def result_gate(r):
 rows=r.get('records',[]);done=r.get('complete') is True and len(rows)==8
 return dict(complete=done,raw_pass=done and all(x['passed'] for x in rows),raw_difference_hold=r.get('raw_difference_hold') is True)
def validate_manifest(m):
 assert m['schema']=='attention-kv-capture-v1' and m['complete'] and m['observers_removed'] and len(m['modules'])==8
 assert sum(x['count'] for x in m['counts'])==128 and {s['phase'] for s in m['samples']}=={'prefill','decode'}
 for s in m['samples']:
  assert s['original_calls']==1 and s['return_is_none'] and all(s['inputs_preserved'].values())
  for part in ('pre_inputs','post_inputs','pre_slots','post_slots'):
   for item in s[part].values():assert sha(item['path'])==item['sha256']
 return m['samples']
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',type=Path);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha;samples=validate_manifest(json.loads(Path(a.manifest).read_text()))
 if not a.run_gpu:print(json.dumps(dict(executed=False,scope='Two actual phases x native2/tile2; compact-prefix shared-stride sentinel cache')));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.fstat(fd);t=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').stat();assert(st.st_dev,st.st_ino)==(t.st_dev,t.st_ino)
 assert a.output and a.output.is_absolute();a.output.mkdir(exist_ok=False)
 for env,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/name).mkdir();os.environ[env]=str(a.output/name)
 r=dict(complete=False,gpu_executed=False,manifest_sha=a.manifest_sha,records=[],timing=False,scope='Compact-prefix synthetic caller-owned cache; actual input logical bytes/strides/slots, not full engine backing')
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('285s')));signal.alarm(285)
 try:
  import torch
  from vllm._custom_ops import reshape_and_cache_flash as native
  from tileir_vllm_kv_scatter_v1 import reshape_and_cache_flash as tile
  from norm_runtime_contract_capture_v1.observer import identity
  torch.set_num_threads(4);r['gpu_executed']=True;r['native']=identity(native)
  def load(item):return torch.load(item['path'],map_location='cpu',weights_only=True)
  def restore(item):
   cpu=load(item);d=item['metadata'];base=torch.full((d['storage_nbytes']//cpu.element_size(),),-7,dtype=cpu.dtype,device='cuda');v=torch.as_strided(base,d['shape'],d['stride'],d['storage_offset']);v.copy_(cpu);return base,v
  for s in samples:
   assert r['native']['source_sha256']==s['helper']['source_sha256']
   desc=s['cache_descriptors'];kd,vd=desc['key'],desc['value'];assert kd['storage_ptr']==vd['storage_ptr']
   bs=kd['shape'][1];slots=torch.tensor(s['slots'],dtype=torch.int64);blocks=slots//bs;offs=slots%bs
   nb=int(blocks.max())+1;shape=[nb]+kd['shape'][1:];extent=max(d['storage_offset']+sum((n-1)*st for n,st in zip(shape,d['stride']))+1 for d in (kd,vd));assert extent*2<=384*1024*1024
   initial=torch.full((extent,),-37,dtype=torch.bfloat16)
   def views(base):return [torch.as_strided(base,shape,d['stride'],d['storage_offset']) for d in (kd,vd)]
   ik,iv=views(initial)
   for name,v in [('key_cache',ik),('value_cache',iv)]:v[blocks,offs]=load(s['pre_slots'][name])
   expected=initial.clone();ek,ev=views(expected)
   for name,v in [('key_cache',ek),('value_cache',ev)]:v[blocks,offs]=load(s['post_slots'][name])
   inp={k:restore(item) for k,item in s['pre_inputs'].items()};before={k:base.cpu() for k,(base,v) in inp.items()};args={k:v for k,(base,v) in inp.items()}
   # Captured helper writes are expected to be pure BF16 copies for every selected token.
   assert torch.equal(load(s['post_slots']['key_cache']).view(torch.uint8),load(s['pre_inputs']['key']).view(torch.uint8))
   assert torch.equal(load(s['post_slots']['value_cache']).view(torch.uint8),load(s['pre_inputs']['value']).view(torch.uint8))
   previous={}
   for arm,fn in [('native',native),('tile',tile)]:
    for repeat in range(2):
     backing=initial.to('cuda');kc,vc=views(backing)
     ret=fn(args['key'],args['value'],kc,vc,args['slot_mapping'],s['kv_cache_dtype'],args['k_scale'],args['v_scale']);torch.cuda.synchronize()
     actual=backing.cpu();raw=torch.equal(actual.view(torch.uint8),expected.view(torch.uint8));preserved=all(torch.equal(base.cpu().view(torch.uint8),before[k].view(torch.uint8)) for k,(base,v) in inp.items())
     digest=hashlib.sha256(actual.view(torch.uint8).numpy().tobytes()).hexdigest();repeat_equal=repeat==0 or previous[arm]==digest
     row=dict(phase=s['phase'],arm=arm,repeat=repeat,return_none=ret is None,full_owned_storage_raw_equal=raw,input_full_owned_storage_preserved=preserved,repeat_raw_equal=repeat_equal,original_capacity=kd['shape'][0],replay_capacity=nb,owned_storage_bytes=extent*2,cache_stride=kd['stride'],cache_offsets=[kd['storage_offset'],vd['storage_offset']],shared_backing=kc.untyped_storage().data_ptr()==vc.untyped_storage().data_ptr(),raw_sha256=digest)
     row['passed']=ret is None and raw and preserved and repeat_equal and row['shared_backing'];r['records'].append(row);save()
     if not row['passed']:
      torch.save(actual,a.output/(s['phase']+'_'+arm+str(repeat)+'_actual.pt'));r['raw_difference_hold']=True;save();return 2
     previous[arm]=digest;del backing,kc,vc,actual
   del initial,expected,ik,iv,ek,ev,inp,args,before
  r['complete']=True;save();return 0
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
