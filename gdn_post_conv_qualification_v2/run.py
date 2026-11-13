"""Frozen actual public post-conv replay; CPU default, no timing."""
import argparse,json,os,sys,signal,hashlib,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
NAMES=('q','k','v','g','beta')
def result_gate(r):
 rows=r.get('records',[])
 complete=r.get('complete') is True and r.get('gpu_executed') is True and len(rows)==4
 passed=complete and all(x.get('passed') is True for x in rows)
 return dict(complete=complete,raw_pass=passed,raw_difference_hold=r.get('raw_difference_hold') is True and not passed)
def validate_manifest(m):
 assert m['schema']=='gdn-post-conv-capture-v1' and m['complete'] and m['observers_removed']
 assert len(m['counts'])==24 and all(x==1 for x in m['counts'].values())
 s,=m['samples'];assert s['phase']['phase']=='prefill' and all(s['input_storage_preserved'].values())
 assert s['arguments']==dict(num_k_heads=16,head_k_dim=128,head_v_dim=128,apply_l2norm=True,output_g_exp=False)
 assert set(s['outputs'])==set(NAMES)
 for part in ('pre','outputs'):
  for item in s[part].values():
   assert sha(item['path'])==item['sha256'] and sha(item['storage_path'])==item['storage_sha256']
 return s
def compare(ref,out,torch):
 assert ref.shape==out.shape and ref.dtype==out.dtype
 x=ref.float();y=out.float();finite=torch.isfinite(x)&torch.isfinite(y);diff=(x-y).abs()
 return dict(raw_equal=torch.equal(ref.contiguous().view(torch.uint8),out.contiguous().view(torch.uint8)),raw_different_elements=int((ref.contiguous().view(torch.int16 if ref.dtype==torch.bfloat16 else torch.int32)!=out.contiguous().view(torch.int16 if out.dtype==torch.bfloat16 else torch.int32)).sum()),reference_nonfinite=int((~torch.isfinite(x)).sum()),candidate_nonfinite=int((~torch.isfinite(y)).sum()),max_abs_finite=float(diff[finite].max()) if finite.any() else None,strict_failures=int(((diff>1e-5+1e-5*x.abs())|~finite).sum()),strict_atol=1e-5,strict_rtol=1e-5)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',type=Path);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 assert sha(a.manifest)==a.manifest_sha;verify_pins(R/'CPU-PINS.json');s=validate_manifest(json.loads(Path(a.manifest).read_text()))
 if not a.run_gpu:print(json.dumps(dict(executed=False,scope='one frozen actual prefill, native2 then TileIR2, no timing')));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);actual=os.fstat(fd);expected=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').stat();assert (actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino)
 assert a.output and a.output.is_absolute();a.output.mkdir(exist_ok=False)
 for env,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/name).mkdir();os.environ[env]=str(a.output/name)
 r=dict(complete=False,gpu_executed=False,manifest_sha=a.manifest_sha,records=[],timing=False,scope='one captured prefill boundary; no model/optimized/decode equivalence claim')
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('285s child bound')));signal.alarm(285)
 try:
  import torch
  from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep as native
  from tileir_vllm_post_conv_v2 import fused_post_conv_prep as tile
  from norm_runtime_contract_capture_v1.observer import metadata,aliases,identity
  from gdn_post_conv_runtime_capture_v1.observer import fullbytes,digest
  torch.set_num_threads(4);r['gpu_executed']=True;r['native_callable']=identity(native)
  assert r['native_callable']['source_sha256']==s['helper']['source_sha256']
  flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
  r['initial_flags']=flags()
  def load(item):return torch.load(item['path'],map_location='cpu',weights_only=True)
  def restore(item):
   m=item['metadata'];cpu=load(item);raw=torch.load(item['storage_path'],map_location='cpu',weights_only=True)
   assert hashlib.sha256(raw.numpy().tobytes()).hexdigest()==item['storage_byte_sha256']
   backing=raw.to('cuda');typed=backing.view(cpu.dtype);v=torch.as_strided(typed,m['shape'],m['stride'],m['storage_offset'])
   assert torch.equal(v.cpu().contiguous().view(torch.uint8),cpu.contiguous().view(torch.uint8))
   return v
  expected_outputs={k:load(v) for k,v in s['outputs'].items()};inputs={k:restore(v) for k,v in s['pre'].items()}
  r['restored_inputs']={k:metadata(v) for k,v in inputs.items()}
  before={k:digest(v) for k,v in inputs.items()};previous={}
  assert all(torch.isfinite(v).all() for v in inputs.values())
  for arm,fn in [('native',native),('tile',tile)]:
   for repeat in range(2):
    row=dict(arm=arm,repeat=repeat,passed=False);r['records'].append(row);save()
    outputs=fn(**inputs,**s['arguments']);torch.cuda.synchronize()
    assert isinstance(outputs,tuple) and len(outputs)==5
    outputs=dict(zip(NAMES,outputs));row['return_kind']='tuple';row['output_order']=list(NAMES)
    row['descriptors']={k:metadata(v) for k,v in outputs.items()}
    row['aliases']=aliases(dict(inputs,**outputs))
    assert not any(v['storage_overlap'] for k,v in row['aliases'].items() if any(n in k.split('|') for n in NAMES))
    for k,v in outputs.items():
     old=s['outputs'][k]['metadata'];new=row['descriptors'][k]
     assert all(new[f]==old[f] for f in ('shape','stride','storage_offset','dtype','storage_nbytes'))
    row['input_fullstorage_preserved']={k:digest(v)==before[k] for k,v in inputs.items()};assert all(row['input_fullstorage_preserved'].values())
    cpu={k:v.cpu() for k,v in outputs.items()};row['comparisons']={k:compare(expected_outputs[k],v,torch) for k,v in cpu.items()}
    row['repeat_raw_equal']=None if repeat==0 else all(torch.equal(previous[arm][k].view(torch.uint8),v.view(torch.uint8)) for k,v in cpu.items())
    passed=all(v['raw_equal'] and v['candidate_nonfinite']==0 and v['reference_nonfinite']==0 for v in row['comparisons'].values()) and row['repeat_raw_equal'] is not False
    row['passed']=passed
    if not passed:
     row['saved_outputs']={}
     for k,v in cpu.items():
      path=a.output/(arm+str(repeat)+'_'+k+'.pt');torch.save(v,path);row['saved_outputs'][k]={'path':str(path),'sha256':sha(path)}
     r.update(raw_difference_hold=True,halted_at=dict(arm=arm,repeat=repeat),final_flags=flags());assert r['final_flags']==r['initial_flags'];save();return 2
    previous[arm]=cpu;del outputs;save()
  r.update(complete=True,final_flags=flags());assert r['final_flags']==r['initial_flags'];verify_pins(R/'CPU-PINS.json');save();return 0
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
