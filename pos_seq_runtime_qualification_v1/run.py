"""Actual2cases x native/tile x2, entire storage correctness only."""
import argparse,fcntl,json,os,signal,sys,traceback,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from pos_seq_runtime_qualification_v1.contract import sha,load_record,verify_case,NAMES
from packed_grouped_qualification_v1.contract import verify_pins
from norm_runtime_qualification_v2.run import identity,aliases,descriptor

def result_gate(r):
 rows=r.get('records',[]);ok=r.get('complete') is True and len(rows)==8 and all(x['passed'] for x in rows)
 return dict(raw_pass=ok,complete=ok,contract_pass=ok,raw_difference_hold=False)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 verify_pins(R/'PINS.json');assert sha(a.manifest)==a.manifest_sha;m=json.loads(a.manifest.read_text());assert m['complete'] and {s['phase'] for s in m['samples']}=={'prefill','decode'} and len(m['samples'])==2
 cpu=[verify_case(s) for s in m['samples']]
 if not a.run_gpu:print(json.dumps(dict(executed=False,cpu=cpu)));return 0
 lock='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock';fd=int(os.environ['ALL_GEMM_LOCK_FD']);assert os.fstat(fd).st_ino==os.stat(lock).st_ino;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for k,n in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/n).mkdir();os.environ[k]=str(a.output/n)
 r=dict(complete=False,gpu_executed=False,manifest_sha=a.manifest_sha,cpu=cpu,records=[],timing=False,scope='Observed regular-device input placement; full storage same-input comparison')
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.alarm(175)
 try:
  import torch
  from vllm.v1.worker.gpu.input_batch import prepare_pos_seq_lens as native
  from tileir_vllm_pos_seq_v1 import prepare_pos_seq_lens as tile
  torch.set_num_threads(4)
  for s in m['samples']:
   assert identity(native)==s['source']['helper']
   expected={k:load_record(s['post'][k])[0] for k in NAMES};repeat_hash={}
   for branch,fn in [('native',native),('tileir',tile)]:
    for repeat in range(2):
     buffers={};values={}
     for k in NAMES:
      raw,view=load_record(s['pre'][k]);d=s['pre'][k]['metadata'];assert d['device'].startswith('cuda') and d['stride']==[1]
      buffers[k]=raw.to('cuda');values[k]=torch.empty(0,dtype=view.dtype,device='cuda').set_(buffers[k].untyped_storage(),d['storage_offset'],d['shape'],d['stride'])
     r['gpu_executed']=True;save();before_alias=aliases(values);before_desc={k:descriptor(v) for k,v in values.items()}
     with torch.inference_mode():ret=fn(*(values[k] for k in NAMES))
     torch.cuda.synchronize();after={k:b.cpu() for k,b in buffers.items()};matches={k:bool(torch.equal(after[k],expected[k])) for k in NAMES};digests={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in after.items()}
     same_repeat=repeat==0 or digests==repeat_hash[branch]
     if repeat==0:repeat_hash[branch]=digests
     passed=ret is None and all(matches.values()) and aliases(values)==before_alias==s['pre_aliases']==s['post_aliases'] and before_desc=={k:descriptor(v) for k,v in values.items()} and same_repeat
     row=dict(phase=s['phase'],branch=branch,repeat=repeat,full_storage_equal=matches,return_none=ret is None,repeat_equal=same_repeat,passed=passed,storage_digests=digests);r['records'].append(row);save();assert passed,'Full storage contract mismatch'
  verify_pins(R/'PINS.json');r['complete']=True;r['gate']=result_gate(r);save();return 0
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
