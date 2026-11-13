"""CPU-default two actual banks: native binding, old repro, new candidate. No model."""
import argparse,json,os,fcntl,signal,sys,hashlib,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from gdn_decode_core_runtime_v3.adapter import gpu_metrics,numerical_pass,raw_equal

def result_gate(r):
 rows=r.get('records',[]);complete=r.get('complete')is True and len(rows)==6
 contract=complete and all(x.get('contract_pass')is True for x in rows)
 raw=contract and all(x.get('passed')is True for x in rows)
 return dict(complete=complete,contract_pass=contract,raw_pass=raw,raw_difference_hold=contract and not raw)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 verify_pins(R/'CPU-PINS.json');assert a.manifest.resolve()==(R/'BANKS.json').resolve()and sha(a.manifest)==a.manifest_sha;banks=json.loads(a.manifest.read_text())
 for bank in banks:
  for item in [*bank['inputs'].values(),*[bank[k]for k in ('before','native_state','native_output','old_state','old_output')]]:assert sha(item['path'])==item['sha256']
 if not a.run_gpu:print(json.dumps(dict(gpu_executed=False,banks=2,calls=6,state_scope='compact33slots remapped; original5GiBowner not reconstructed')));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);lock='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock';assert os.fstat(fd).st_ino==os.stat(lock).st_ino;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for k,d in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TRITON_CACHE_DIR','triton'),('TMPDIR','tmp')]:
  (a.output/d).mkdir();os.environ[k]=str(a.output/d)
 r=dict(complete=False,gpu_executed=False,manifest_sha=a.manifest_sha,records=[],timing=False,scope='two actualbanks,compactselectedstate and slotremap; native/oldrepro prerequisites')
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.alarm(285)
 try:
  import torch
  from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode as native
  from tileir_gdn_decode_core_v3 import packed_decode as old
  from tileir_gdn_decode_core_v4 import packed_decode as new
  torch.set_num_threads(4);r['gpu_executed']=True
  def load(x):return torch.load(x['path'],weights_only=True,map_location='cpu')
  for bank in banks:
   cpu={k:load(x)for k,x in bank['inputs'].items()};before=load(bank['before']);expected_state=load(bank['native_state']);expected_out=load(bank['native_output']);oldstate=load(bank['old_state']);oldout=load(bank['old_output'])
   slots=bank['selected_slots'];lookup={s:i+1 for i,s in enumerate(slots)};mapped=[lookup[s]for s in cpu['ssm_state_indices'].tolist()];assert len(slots)==32 and len(set(mapped))==32
   for label,fun in [('native',native),('old',old),('new',new)]:
    row=dict(bank=bank['name'],backend=label,contract_pass=False,passed=False);r['records'].append(row);save()
    inp={}
    for k,t in cpu.items():
     stride=bank['inputs'][k]['metadata']['stride'];x=torch.empty_strided(t.shape,stride,dtype=t.dtype,device='cuda');x.copy_(t);inp[k]=x
    inp['ssm_state_indices'].copy_(torch.tensor(mapped,dtype=torch.int32,device='cuda'));snap={k:v.clone()for k,v in inp.items()}
    state=torch.full((33,32,128,128),-7.,dtype=torch.float32,device='cuda');state[1:].copy_(before);out=torch.empty(expected_out.shape,dtype=expected_out.dtype,device='cuda')
    got=fun(**inp,scale=128**-0.5,initial_state=state,out=out,use_qk_l2norm_in_kernel=True);torch.cuda.synchronize()
    refout=oldout if label=='old'else expected_out;refstate=oldstate if label=='old'else expected_state
    row['comparisons']=dict(output=gpu_metrics(out,refout.to('cuda'),torch),state=gpu_metrics(state[1:],refstate.to('cuda'),torch))
    row['contract_pass']=isinstance(got,tuple)and len(got)==2 and got[0]is out and got[1]is state and bool((state[0]==-7).all().item())and all(raw_equal(inp[k],snap[k],torch)for k in inp)
    artifacts={}
    for key,t in [('output',out),('selected_state',state[1:])]:
     path=a.output/(bank['name']+'-'+label+'-'+key+'.pt');torch.save(t.detach().cpu().contiguous(),path);artifacts[key]=dict(path=str(path),sha256=sha(path))
    row['artifacts']=artifacts;row['passed']=row['contract_pass']and numerical_pass(row['comparisons']);save()
    assert row['contract_pass'],'storage/input/tuple HOLD'
    if label in('native','old'):assert row['passed'],'saved native/old reproduction HOLD; no next launch'
    if label=='new'and not row['passed']:raise AssertionError('new candidate numerical HOLD; no next bank')
    del got,inp,snap,state,out
  r['complete']=True;save()
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
 return 0 if result_gate(r)['raw_pass']else 2
if __name__=='__main__':raise SystemExit(main())
