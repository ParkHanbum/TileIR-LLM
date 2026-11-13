"""Actual saved GDN KKT stage replay. No runtime replacement or timing."""
import argparse,hashlib,json,os,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
NAMES=('k','beta','cu_seqlens','chunk_indices')
CUMSUM=P/'gdn_chunk_cumsum_qualification_v1/gdnCumsumQual01/native_reference.pt'
BASE=P/'gdn_chunk_kkt_qualification_v1/gdnKKTQual01'
def result_gate(r):
 rows=r.get('records',[])
 complete=r.get('complete') is True and r.get('gpu_executed') is True and len(rows)==2
 passed=complete and r.get('native_baseline_verified') is True and all(x.get('passed') is True for x in rows) and [(x.get('arm'),x.get('repeat')) for x in rows]==[('tile',0),('tile',1)] and r.get('final_flags')==r.get('initial_flags')==[True,True]
 return dict(complete=complete,raw_pass=passed,raw_difference_hold=r.get('raw_difference_hold') is True and not passed)
def baseline():
 r=json.loads((BASE/'RESULT.json').read_text());rows=r['records']
 assert r['complete'] is False and r['gpu_executed'] is True and len(rows)==3
 assert [(x['arm'],x['repeat']) for x in rows]==[('native',0),('native',1),('tile',0)]
 for x in rows[:2]:
  c=x['comparison'];assert x['passed'] and c['raw_equal'] and c['strict_failures']==c['reference_nonfinite']==c['candidate_nonfinite']==0
  assert c['elements']==3473408 and x['fresh_disjoint'] and all(x['inputs_preserved'].values()) and x['masked_positive_zero']
 assert rows[1]['repeat_raw_equal'] is True and not rows[2]['passed'] and 'comparison' not in rows[2]
 assert 'bitcast' in r['error'] and '32' in r['error'] and '19' in r['error']
 assert rows[0]['actual_autotuner_winner']==rows[1]['actual_autotuner_winner']
 selected=rows[0]['actual_autotuner_winner'];assert selected['kwargs']=={'BK':64} and selected['num_warps']==8 and selected['num_stages']==3
 assert r['initial_flags']==[True,True]
 assert sha(BASE/'native_reference.pt')==r['native_reference_sha']
 return r,selected
def validate_manifest(m):
 assert m['schema']=='gdn-chunk-capture-v1' and m['complete'] is True and m['observers_removed'] is True and not m['errors']
 assert len(m['counts'])==24 and all(v==1 for v in m['counts'].values())
 s,=m['samples'];assert s['helper']['qualname']=='ChunkGatedDeltaRule.forward_native'
 assert all(s['input_logical_preserved'].values())
 assert s['arguments']==dict(output_final_state=True,use_qk_l2norm_in_kernel=False,core_attn_out=None,aiter_prefill_metadata=None)
 for key in NAMES:
  item=s['pre'][key];assert sha(item['path'])==item['sha256']
 return s

def compare(ref,out,torch):
 assert ref.shape==out.shape and ref.dtype==out.dtype
 diff=(ref-out).abs();finite=torch.isfinite(ref)&torch.isfinite(out)
 return dict(elements=ref.numel(),raw_equal=torch.equal(ref.view(torch.uint8),out.view(torch.uint8)),raw_different_elements=int((ref.view(torch.int32)!=out.view(torch.int32)).sum()),reference_nonfinite=int((~torch.isfinite(ref)).sum()),candidate_nonfinite=int((~torch.isfinite(out)).sum()),max_abs_finite=float(diff[finite].max()) if finite.any() else None,strict_failures=int(((diff>1e-5+1e-5*ref.abs())|~finite).sum()),strict_atol=1e-5,strict_rtol=1e-5)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',type=Path);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 assert sha(a.manifest)==a.manifest_sha;verify_pins(R/'CPU-PINS.json');s=validate_manifest(json.loads(Path(a.manifest).read_text()));prior,selected=baseline();assert prior['manifest_sha']==a.manifest_sha
 if not a.run_gpu:print(json.dumps(dict(executed=False,scope='actual KKT stage, saved native2 reference then TileIR2, no timing')));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);actual=os.fstat(fd);expected=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').stat();assert (actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino)
 assert a.output and a.output.is_absolute();a.output.mkdir(exist_ok=False)
 for env,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton')]:
  (a.output/name).mkdir();os.environ[env]=str(a.output/name)
 r=dict(complete=False,gpu_executed=False,manifest_sha=a.manifest_sha,records=[],timing=False,scope='one actual KKT stage, native cumsum and previous KKT reference reused, not whole GDN/model equivalence')
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('285s child bound')));signal.alarm(285)
 try:
  import torch
  from tileir_gdn_chunk_kkt_v3 import chunk_scaled_dot_kkt as tile
  from tileir_gdn_chunk_cumsum_v1.contract import validate_metadata
  from norm_runtime_contract_capture_v1.observer import metadata,identity
  torch.set_num_threads(4);r['gpu_executed']=True;r['native_baseline_verified']=True
  r['native_baseline']=dict(result=str(BASE/'RESULT.json'),result_sha=sha(BASE/'RESULT.json'),reference=str(BASE/'native_reference.pt'),reference_sha=sha(BASE/'native_reference.pt'),actual_previous_winner=selected,scope='Prior process native2 output, not a fresh same-run native reference')
  flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
  r['initial_flags']=flags();assert r['initial_flags']==[True,True];cpus={}
  for key in NAMES:
   item=s['pre'][key];v=torch.load(item['path'],map_location='cpu',weights_only=True)
   assert hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==item['logical_sha256']
   m=item['metadata'];assert tuple(v.shape)==tuple(m['shape']) and tuple(v.stride())==tuple(m['stride']) and v.storage_offset()==m['storage_offset']==0 and str(v.dtype)==m['dtype']
   assert v.untyped_storage().nbytes()==v.numel()*v.element_size()==m['storage_nbytes']
   cpus[key]=v
  validate_metadata(cpus['cu_seqlens'].tolist(),cpus['chunk_indices'].tolist())
  cpus['g']=torch.load(CUMSUM,map_location='cpu',weights_only=True)
  assert tuple(cpus['g'].shape)==(1,1696,32) and cpus['g'].dtype==torch.float32 and cpus['g'].is_contiguous()
  assert all(torch.isfinite(cpus[n]).all() for n in ('k','beta','g'))
  r['cumsum_reference']=dict(path=str(CUMSUM),sha256=sha(CUMSUM))
  inputs={k:v.to('cuda') for k,v in cpus.items()};r['input_descriptors']={k:metadata(v) for k,v in inputs.items()}
  reference=torch.load(BASE/'native_reference.pt',map_location='cpu',weights_only=True);previous={}
  assert tuple(reference.shape)==(1,1696,32,64) and reference.dtype==torch.float32 and reference.is_contiguous()
  for arm,fn in [('tile',tile)]:
   for repeat in range(2):
    row=dict(arm=arm,repeat=repeat,passed=False);r['records'].append(row);save()
    row['candidate_BK']=selected['kwargs']['BK']
    out=fn(**inputs,bk=row['candidate_BK']);torch.cuda.synchronize()
    assert isinstance(out,torch.Tensor) and out.dtype==torch.float32 and tuple(out.shape)==(1,1696,32,64) and out.is_cuda
    assert out.stride()==(3473408,2048,64,1) and out.storage_offset()==0 and out.untyped_storage().nbytes()==13893632
    row['output_descriptor']=metadata(out);row['fresh_disjoint']=all(out.untyped_storage().data_ptr()!=v.untyped_storage().data_ptr() for v in inputs.values());assert row['fresh_disjoint']
    row['inputs_preserved']={k:torch.equal(v.cpu().view(torch.uint8),cpus[k].view(torch.uint8)) for k,v in inputs.items()};assert all(row['inputs_preserved'].values())
    cpu=out.cpu()
    row_index=torch.arange(1696)%53;col_index=torch.arange(64)
    allowed=(row_index[:,None]>col_index[None,:]).reshape(1,1696,1,64).expand_as(cpu)
    row['masked_positive_zero']=bool((cpu.masked_select(~allowed).view(torch.int32)==0).all())
    assert row['masked_positive_zero']
    row['comparison']=compare(reference,cpu,torch)
    row['repeat_raw_equal']=None if repeat==0 else torch.equal(previous[arm].view(torch.uint8),cpu.view(torch.uint8))
    c=row['comparison'];row['passed']=c['raw_equal'] and c['strict_failures']==0 and c['reference_nonfinite']==0 and c['candidate_nonfinite']==0 and row['repeat_raw_equal'] is not False
    if not row['passed']:
     path=a.output/(arm+str(repeat)+'.pt');torch.save(cpu,path);row['saved_output']=dict(path=str(path),sha256=sha(path));r.update(raw_difference_hold=True,halted_at=dict(arm=arm,repeat=repeat),final_flags=flags());assert r['final_flags']==r['initial_flags'];save();return 2
    previous[arm]=cpu;del out;save()
  r.update(complete=True,final_flags=flags());assert r['final_flags']==r['initial_flags'];verify_pins(R/'CPU-PINS.json');save();return 0
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
