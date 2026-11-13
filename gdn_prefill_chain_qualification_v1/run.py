"""One-bank six-stage native2/TileIR2 chain qualification, correctness only."""
from pathlib import Path
import argparse,hashlib,json,os,signal,sys,importlib,shutil,traceback
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from gdn_post_conv_qualification_v2.run import compare
from gdn_prefill_stage_capture_v1.observer import Observer,STAGES
from gdn_prefill_chain_qualification_v1.chain import execute,LABELS
C=P/'gdn_prefill_stage_capture_v1';B=C/'gdnStagesCapture01'
class RawHold(Exception):pass
def result_gate(r):
 rows=r.get('records',[]);order=[(a,i) for a in ('native','tile') for i in (0,1)]
 passed=r.get('complete') is True and [(x['arm'],x['repeat']) for x in rows]==order and all(x.get('passed') is True for x in rows) and r.get('initial_flags')==r.get('final_flags')==[True,True]
 return dict(complete=r.get('complete') is True,raw_pass=passed,raw_difference_hold=r.get('raw_difference_hold') is True and not passed)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 assert sha(a.manifest)==a.manifest_sha;verify_pins(R/'CPU-PINS.json');m=json.loads(Path(a.manifest).read_text());s,=m['samples'];cap=json.loads((B/'RESULT.json').read_text())
 assert m['complete'] and m['observers_removed'] and all(s['input_logical_preserved'].values()) and cap['complete'] and cap['artifacts_written']
 assert s['arguments']==dict(output_final_state=True,use_qk_l2norm_in_kernel=False,core_attn_out=None,aiter_prefill_metadata=None)
 for e in [*s['pre'].values(),*s['outputs'].values(),*cap['artifacts'].values()]:assert sha(e['path'])==e['sha256']
 if not a.run_gpu:print('CPU pins/actual bank PASS, no execution');return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.fstat(fd);ls=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').stat();assert(st.st_dev,st.st_ino)==(ls.st_dev,ls.st_ino)
 a.output.mkdir(exist_ok=False)
 for env,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/name).mkdir();os.environ[env]=str(a.output/name)
 r=dict(complete=False,gpu_executed=False,manifest_sha=a.manifest_sha,records=[],timing=False,scope='actual layer0 bank, fresh native2 vs full TileIR chain2, no model/24layer claim',first_difference=None)
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('885s chain bound')));signal.alarm(885)
 try:
  from gdn_prefill_stage_capture_v1.cache_groups import relocate
  cp=json.loads((C/'CACHE-PINS.json').read_text());srcroot=Path(cp['source'])
  for rel,h in cp['files'].items():
   src=srcroot/rel;assert not src.is_symlink() and sha(src)==h;dst=a.output/'triton'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(dst)==h
  for p in sorted((a.output/'triton').rglob('__grp__*.json')):
   names=[f.name for f in p.parent.iterdir() if f.is_file() and f!=p];p.write_text(json.dumps(relocate(json.loads(p.read_text()),p.parent,names)))
  import torch
  from norm_runtime_contract_capture_v1.observer import identity
  from tileir_gdn_chunk_cumsum_v1 import chunk_local_cumsum
  from tileir_gdn_chunk_kkt_v5 import chunk_scaled_dot_kkt
  from tileir_gdn_solve_tril_v5 import solve_tril
  from tileir_gdn_wy_v1 import recompute_w_u_fwd
  from tileir_gdn_chunk_state_v1 import chunk_state
  from tileir_gdn_chunk_output_v2 import chunk_output
  ops=dict(cumsum=chunk_local_cumsum,kkt=chunk_scaled_dot_kkt,solve=solve_tril,wy=recompute_w_u_fwd,state=chunk_state,output=chunk_output)
  chunk=importlib.import_module('vllm.third_party.flash_linear_attention.ops.chunk');solver=importlib.import_module('vllm.third_party.flash_linear_attention.ops.solve_tril');assert solver.FLA_TRIL_PRECISION=='ieee' and solver.is_tma_supported is False
  r.update(actual_solver_precision=solver.FLA_TRIL_PRECISION,actual_solver_USE_TMA=solver.is_tma_supported,gpu_executed=True);torch.set_num_threads(4)
  flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
  r['initial_flags']=flags();assert r['initial_flags']==[True,True]
  def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  def load(e):
   v=torch.load(e['path'],map_location='cpu',weights_only=True);assert digest(v)==e['logical_sha256'];return v
  cpus={k:load(e) for k,e in s['pre'].items()};expected={k:load(e) for k,e in cap['artifacts'].items()};full={k:load(e) for k,e in s['outputs'].items()}
  for k,v in cpus.items():
   d=s['pre'][k]['metadata'];assert list(v.shape)==d['shape'] and list(v.stride())==d['stride'] and str(v.dtype)==d['dtype'] and v.storage_offset()==0 and v.is_contiguous() and v.untyped_storage().nbytes()==v.numel()*v.element_size()==d['storage_nbytes']<=128*1024*1024
  assert set(expected)==set(LABELS);assert all(torch.isfinite(v).all() for v in list(cpus.values())+list(expected.values()));prior={};native_final=None
  def overlap(a,b):
   aa=a.untyped_storage();bb=b.untyped_storage();return aa.data_ptr()<bb.data_ptr()+bb.nbytes() and bb.data_ptr()<aa.data_ptr()+aa.nbytes()
  with torch.inference_mode():
   for arm in ('native','tile'):
    for repeat in (0,1):
     row=dict(arm=arm,repeat=repeat,passed=False,stages=[]);r['records'].append(row);save();inputs={k:v.cuda() for k,v in cpus.items()};retained={};seen=[];obs=None
     def snapshot(t,label):
      assert label not in seen and label in expected and isinstance(t,torch.Tensor) and t.is_cuda
      e=expected[label];assert t.shape==e.shape and t.dtype==e.dtype and t.stride()==e.stride() and t.storage_offset()==0 and t.untyped_storage().nbytes()==t.numel()*t.element_size()<=128*1024*1024
      torch.cuda.synchronize();cpu=t.cpu();metric=compare(e,cpu,torch);fresh=all(not overlap(t,v) for v in [*inputs.values(),*retained.values()]);preserved=all(digest(v)==s['pre'][k]['logical_sha256'] for k,v in inputs.items());current=digest(cpu);rep=repeat==0 or prior[(arm,label)]==current
      prior_preserved=all(digest(v)==cap['artifacts'][k]['logical_sha256'] for k,v in retained.items())
      check=dict(label=label,comparison=metric,fresh_disjoint=fresh,inputs_preserved=preserved,prior_stages_preserved=prior_preserved,repeat_raw=rep)
      check['passed']=metric['raw_equal'] and metric['strict_failures']==metric['reference_nonfinite']==metric['candidate_nonfinite']==0 and fresh and preserved and prior_preserved and rep;row['stages'].append(check);seen.append(label);retained[label]=t;save()
      if not check['passed']:
       assert r['first_difference'] is None;r['first_difference']=dict(arm=arm,repeat=repeat,label=label);r['raw_difference_hold']=True;torch.save(cpu,a.output/'first_difference.pt');save();raise RawHold(label)
      prior[(arm,label)]=current;return check
     def observe(outputs):
      for label,t in outputs.items():snapshot(t,label)
     try:
      if arm=='native':
       obs=Observer(chunk,snapshot,identity);obs.install()
       try:out=chunk.chunk_gated_delta_rule(**inputs,output_final_state=True,use_qk_l2norm_in_kernel=False,core_attn_out=None)
       finally:obs.close();row['observers_restored']=obs.closed
       assert obs.complete();row['native_helper_counts']=obs.counts
      else:out=execute(inputs,ops,observe)
      assert seen==list(LABELS) and isinstance(out,tuple) and len(out)==2
      row['full_output_comparisons']={name:compare(full[name],value.cpu(),torch) for name,value in zip(('o','final_state'),out)}
      assert all(c['raw_equal'] and c['strict_failures']==c['reference_nonfinite']==c['candidate_nonfinite']==0 for c in row['full_output_comparisons'].values())
      fdig=tuple(digest(t) for t in out)
      if arm=='native':native_final=fdig
      else:assert fdig==native_final
      assert all(digest(v)==s['pre'][k]['logical_sha256'] for k,v in inputs.items());row['passed']=True;save()
     finally:
      if obs is not None and obs.bindings:obs.close()
     del retained,inputs,out
  r['final_flags']=flags();assert r['final_flags']==r['initial_flags'];r['complete']=True;verify_pins(R/'CPU-PINS.json');save();return 0
 except RawHold:r['raw_difference_hold']=True;save();return 2
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0);save()
if __name__=='__main__':raise SystemExit(main())
