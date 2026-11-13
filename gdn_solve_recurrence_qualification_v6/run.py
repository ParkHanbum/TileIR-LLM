"""Public-source observational copies, not uninstrumented internal-state evidence."""
from pathlib import Path
import argparse,json,os,sys,signal,traceback,hashlib,shutil
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from gdn_post_conv_qualification_v2.run import compare
LABELS=['d1','d2','d3','d4','x21','x32','x43','x31','x42','x41']
class Hold(Exception):pass
def result_gate(r):
 return dict(raw_pass=r.get('complete') is True and r.get('bindings_pass') is True and len(r.get('banks',[]))==2 and all(b.get('candidate_pass') is True for b in r['banks']),raw_difference_hold=r.get('hold') is True)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 assert sha(a.manifest)==a.manifest_sha;verify_pins(R/'CPU-PINS.json');m=json.loads(Path(a.manifest).read_text())
 for path,h in m['files'].items():assert sha(path)==h
 pre=json.loads((R/'PRESEED.json').read_text())
 for rel,h in pre['files'].items():assert sha(Path(pre['source'])/rel)==sha(Path(pre['destination'])/rel)==h
 if not a.run_gpu:print('CPU source/bank pins PASS; GPU 0');return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.fstat(fd);ls=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').stat();assert(st.st_dev,st.st_ino)==(ls.st_dev,ls.st_ino)
 a.output.mkdir(exist_ok=False)
 for env,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/name).mkdir();os.environ[env]=str(a.output/name)
 r=dict(complete=False,manifest_sha=a.manifest_sha,banks=[],bindings_pass=False,candidate_v6=True,timing=False)
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('885s bound')));signal.alarm(885)
 try:
  from gdn_prefill_stage_capture_v1.cache_groups import relocate
  for rel,h in pre['files'].items():
   src=Path(pre['destination'])/rel;dst=a.output/'triton'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(dst)==h
  for group in (a.output/'triton').rglob('__grp__*.json'):
   names=[f.name for f in group.parent.iterdir() if f.is_file() and f!=group];group.write_text(json.dumps(relocate(json.loads(group.read_text()),group.parent,names)))
  r['preseed']=dict(receipt_sha=sha(R/'PRESEED.json'),count=pre['count'],historical_config=pre['historical_config']);save()
  import torch,cuda.tile as ct,importlib
  from tileir_gdn_solve_tril_v6 import solve_tril as candidate
  solver=importlib.import_module('vllm.third_party.flash_linear_attention.ops.solve_tril');assert solver.FLA_TRIL_PRECISION=='ieee' and solver.is_tma_supported is False
  # Public Python heuristic wrapper -> public Autotuner object. No private ABI or hooks.
  auto=solver.merge_16x16_to_64x64_inverse_kernel.fn
  assert type(auto).__name__=='Autotuner' and hasattr(auto,'cache')
  def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  def flags():return [torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
  r['initial_flags']=flags();assert flags()==[True,True]
  cu=torch.arange(33,dtype=torch.int32,device='cuda')*53;ci=torch.stack((torch.arange(32,dtype=torch.int32,device='cuda'),torch.zeros(32,dtype=torch.int32,device='cuda')),1)
  def valid(metric):return metric['raw_equal'] and metric['strict_failures']==metric['reference_nonfinite']==metric['candidate_nonfinite']==0
  with torch.inference_mode():
   for name,bank in m['banks'].items():
    row=dict(name=name,bindings=[],winner=None,intermediate_comparisons={});r['banks'].append(row);save()
    cpu=torch.load(bank['A'],map_location='cpu',weights_only=True);A=cpu.cuda();refs={k:torch.load(bank[k],map_location='cpu',weights_only=True) for k in ('native','own')};before=digest(A);metadata_before=[digest(cu),digest(ci)]
    def check(value,ref,tag):
     torch.cuda.synchronize();metric=compare(ref,value.cpu(),torch);preserved=digest(A)==before and [digest(cu),digest(ci)]==metadata_before
     record=dict(tag=tag,metric=metric,inputs_preserved=preserved);row['bindings'].append(record);save()
     if not(valid(metric) and preserved):
      failure=a.output/(name+'-'+tag+'-mismatch.pt');torch.save(value.detach().cpu(),failure);r.update(hold=True,reason='Final output binding failed',failed_tag=tag,mismatch_file=str(failure),mismatch_sha=sha(failure));save();raise Hold(tag)
    for rep in range(2):
     out=solver.solve_tril(A,cu,ci,output_dtype=torch.bfloat16)
     config=auto.best_config
     winner=dict(kwargs=config.kwargs,num_warps=config.num_warps,num_stages=config.num_stages,num_ctas=config.num_ctas,maxnreg=config.maxnreg)
     matched=[repr(k) for k,v in auto.cache.items() if v==config]
     assert matched,'actual selected best_config must also be present in live public cache'
     if row['winner'] is None:row['winner']=winner;row['matching_live_cache_keys']=matched
     else:assert row['winner']==winner
     save()
     if winner!=pre['historical_config']:
      failure=a.output/(name+f'-native{rep}-config-hold.pt');torch.save(out.detach().cpu(),failure);r.update(hold=True,reason='Live winner differs from historical config',mismatch_file=str(failure),mismatch_sha=sha(failure));save();raise Hold('config')
     check(out,refs['native'],f'native{rep}');del out
    assert winner['kwargs']=={} and winner['num_ctas']==1
    for rep in range(2):
     out=candidate(A,cu,ci);check(out,refs['native'],f'candidate{rep}')
     if rep==0:prior=digest(out)
     else:assert digest(out)==prior
     del out
    row['candidate_pass']=True;save()
  r['final_flags']=flags();assert r['final_flags']==r['initial_flags'];r.update(complete=True,bindings_pass=True);verify_pins(R/'CPU-PINS.json');save();return 0
 except Hold:r['hold']=True;save();return 2
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0);save()
if __name__=='__main__':raise SystemExit(main())
