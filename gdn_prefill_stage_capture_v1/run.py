"""Full public FLA replay and accepted stage capture only after exact original reconstruction."""
import argparse,os,json,hashlib,signal,sys,traceback,importlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from observer import Observer,STAGES
CUMSUM=P/'gdn_chunk_cumsum_qualification_v1/gdnCumsumQual01/native_reference.pt'
KKT=P/'gdn_chunk_kkt_qualification_v1/gdnKKTQual01/native_reference.pt'
def result_gate(r):
 passed=r.get('complete') is True and r.get('gpu_executed') is True and r.get('full_final_raw') is True and r.get('existing_A_raw_equal') is True and r.get('existing_cumsum_raw_equal') is True and r.get('observers_restored') is True and r.get('stage_counts')=={n:1 for n in STAGES} and r.get('initial_flags')==r.get('final_flags')==[True,True] and r.get('artifacts_written') is True
 return dict(complete=passed,raw_pass=passed,raw_difference_hold=r.get('raw_difference_hold') is True and not passed,scope='native full-function/stage provenance only; no TileIR compatibility claim')
def validate_manifest(m):
 assert m['schema']=='gdn-chunk-capture-v1' and m['complete'] and m['observers_removed'] and not m['errors']
 s,=m['samples'];assert s['helper']['qualname']=='ChunkGatedDeltaRule.forward_native'
 assert all(s['input_logical_preserved'].values()) and s['arguments']==dict(output_final_state=True,use_qk_l2norm_in_kernel=False,core_attn_out=None,aiter_prefill_metadata=None)
 for items in (s['pre'],s['outputs']):
  for item in items.values():assert sha(item['path'])==item['sha256']
 return s

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',type=Path);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 assert sha(a.manifest)==a.manifest_sha;verify_pins(R/'CPU-PINS.json');s=validate_manifest(json.loads(Path(a.manifest).read_text()))
 if not a.run_gpu:print(json.dumps(dict(executed=False,scope='native fullcore replay, exactfinal+A gates before publishing stages')));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);f=os.fstat(fd);l=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').stat();assert (f.st_dev,f.st_ino)==(l.st_dev,l.st_ino)
 assert a.output and a.output.is_absolute();a.output.mkdir(exist_ok=False)
 for key,name in [('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/name).mkdir();os.environ[key]=str(a.output/name)
 r=dict(complete=False,gpu_executed=False,manifest_sha=a.manifest_sha,records=[],timing=False,artifacts_written=False)
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('585s bound')));signal.alarm(585);obs=None
 try:
  import shutil
  from cache_groups import relocate
  cp=json.loads((R/'CACHE-PINS.json').read_text());srcroot=Path(cp['source'])
  for rel,h in cp['files'].items():
   src=srcroot/rel;assert not src.is_symlink() and sha(src)==h
   dst=a.output/'triton'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(dst)==h
  for p in sorted((a.output/'triton').rglob('__grp__*.json')):
   names=[f.name for f in p.parent.iterdir() if f.is_file() and f!=p];p.write_text(json.dumps(relocate(json.loads(p.read_text()),p.parent,names)))
  import torch
  from norm_runtime_contract_capture_v1.observer import metadata,identity,aliases
  from gdn_post_conv_qualification_v2.run import compare
  torch.set_num_threads(4);r['gpu_executed']=True
  flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
  r['initial_flags']=flags();assert r['initial_flags']==[True,True]
  def cpu_load(item):
   x=torch.load(item['path'],map_location='cpu',weights_only=True);assert hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==item['logical_sha256'];return x
  cpus={k:cpu_load(v) for k,v in s['pre'].items()};expected={k:cpu_load(v) for k,v in s['outputs'].items()}
  inputs={}
  for k,x in cpus.items():
   m=s['pre'][k]['metadata'];assert x.is_contiguous() and x.storage_offset()==0 and x.untyped_storage().nbytes()==x.numel()*x.element_size()==m['storage_nbytes']<=128*1024*1024
   assert tuple(x.shape)==tuple(m['shape']) and tuple(x.stride())==tuple(m['stride']) and str(x.dtype)==m['dtype']
   inputs[k]=x.to('cuda')
  assert all(torch.isfinite(x).all() for x in inputs.values())
  r['initial_state_all_zero']=bool((cpus['initial_state']==0).all())
  saved={};total=0
  def snapshot(t,label):
   nonlocal total
   assert isinstance(t,torch.Tensor);n=t.numel()*t.element_size();assert t.untyped_storage().nbytes()<=128*1024*1024 and total+n<=512*1024*1024
   assert label not in saved;total+=n;saved[label]=t.detach().cpu().clone()
   return dict(metadata=metadata(t),logical_bytes=n)
  chunk=importlib.import_module('vllm.third_party.flash_linear_attention.ops.chunk')
  r['full_callable']=identity(chunk.chunk_gated_delta_rule)
  r['settings']={k:os.environ.get(k) for k in ('FLA_TRIL_PRECISION','FLA_USE_FAST_OPS','FLA_SUPPRESS_LEVEL')}
  st=importlib.import_module('vllm.third_party.flash_linear_attention.ops.solve_tril')
  r['actual_solver_precision']=st.FLA_TRIL_PRECISION;r['actual_solver_USE_TMA']=st.is_tma_supported
  obs=Observer(chunk,snapshot,identity);obs.install()
  try:
   with torch.inference_mode():out=chunk.chunk_gated_delta_rule(**inputs,output_final_state=True,use_qk_l2norm_in_kernel=False,core_attn_out=None)
   torch.cuda.synchronize()
  finally:obs.close();r['observers_restored']=obs.closed
  assert isinstance(out,tuple) and len(out)==2 and obs.complete()
  r['stage_counts']=obs.counts;r['stages']=obs.rows;r['artifact_logical_bytes']=total
  r['final_comparisons']={k:compare(expected[k],v.detach().cpu(),torch) for k,v in zip(('o','final_state'),out)}
  r['full_final_raw']=all(c['raw_equal'] and c['strict_failures']==c['reference_nonfinite']==c['candidate_nonfinite']==0 for c in r['final_comparisons'].values())
  r['existing_A_raw_equal']=torch.equal(saved['A_kkt'].view(torch.uint8),torch.load(KKT,map_location='cpu',weights_only=True).view(torch.uint8))
  r['existing_cumsum_raw_equal']=torch.equal(saved['g_cumsum'].view(torch.uint8),torch.load(CUMSUM,map_location='cpu',weights_only=True).view(torch.uint8))
  r['input_preservation']={k:torch.equal(x.cpu().view(torch.uint8),cpus[k].view(torch.uint8)) for k,x in inputs.items()};assert all(r['input_preservation'].values())
  r['output_aliases']=aliases(dict(inputs,output_o=out[0],output_final=out[1]));assert not any(v['storage_overlap'] for k,v in r['output_aliases'].items() if 'output_' in k)
  r['final_flags']=flags();assert r['final_flags']==r['initial_flags']
  # Live public Autotuner observations, no winner inferred from cache directories.
  from triton.runtime.autotuner import Autotuner
  r['autotuners']=[]
  for modname in ('cumsum','chunk_scaled_dot_kkt','solve_tril','wy_fast','chunk_delta_h','chunk_o'):
   mod=importlib.import_module('vllm.third_party.flash_linear_attention.ops.'+modname)
   for name,obj in vars(mod).items():
    if not name.endswith('kernel') and 'kernel_' not in name:continue
    for depth in range(4):
     if isinstance(obj,Autotuner):
      if hasattr(obj,'best_config'):
       c=obj.best_config;r['autotuners'].append(dict(module=modname,name=name,kwargs=dict(c.kwargs),num_warps=c.num_warps,num_stages=c.num_stages,cache_keys=[repr(k) for k in obj.cache]))
      break
     obj=getattr(obj,'fn',None)
     if obj is None:break
  if not (r['full_final_raw'] and r['existing_A_raw_equal'] and r['existing_cumsum_raw_equal']):r['raw_difference_hold']=True;save();return 2
  r['artifacts']={}
  for label,t in saved.items():
   assert torch.isfinite(t).all();p=a.output/(label+'.pt');torch.save(t,p);r['artifacts'][label]=dict(path=str(p),sha256=sha(p),shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),logical_sha256=hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
  r.update(complete=True,artifacts_written=True);verify_pins(R/'CPU-PINS.json');save();return 0
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:
  if obs is not None and obs.bindings:obs.close()
  signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
