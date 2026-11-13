"""One saved actual layer8 gatednorm input; diagnostic only."""
import argparse,fcntl,json,os,sys,signal,traceback,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent.parent;R=Path(__file__).resolve().parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from norm_runtime_qualification_v2.compare import metrics
PROGRESS=P/'gated_norm_runtime_integration_v1/gatedIntegration01/shadow/shadow-PROGRESS.json'
def result_gate(r):
 complete=r.get('complete') is True
 passed=complete and r['inputs_preserved'] and all(r[k]['raw_equal'] for k in ('helper_vs_saved_native','clean_vs_saved_tile','instrumented_vs_clean')) and r['rstd']['all_finite']
 return dict(complete=complete,raw_pass=passed,raw_difference_hold=complete and not passed,scope='Diagnostic bindings only; not native/candidate equality or repair')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-gpu',action='store_true');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();verify_pins(R/'CPU-PINS.json')
 if not a.run_gpu:print('CPU plan: saved layer8 same-stride native/helper/clean/instrumented binding');return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);assert os.fstat(fd).st_ino==os.stat('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').st_ino;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for k,n in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TRITON_CACHE_DIR','triton'),('TMPDIR','tmp')]:
  (a.output/n).mkdir();os.environ[k]=str(a.output/n)
 r=dict(complete=False,gpu_executed=False,source_progress_sha=sha(PROGRESS),timing=False,artifacts={})
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.alarm(175)
 try:
  import torch,cuda.tile as ct
  from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd,calc_rows_per_block
  from tileir_vllm_gated_norm_full_div_v1 import gated_rms_norm
  from gated_first_full_rstd_v1.kernels import probe
  torch.set_num_threads(4);first=json.loads(PROGRESS.read_text())['first_mismatch'];assert first['call_index']==6 and first['path'].endswith('layers.8.linear_attn.norm')
  saved={};descs={}
  for k,item in first['artifacts'].items():
   assert sha(item['path'])==item['sha256'];saved[k]=torch.load(item['path'],map_location='cpu',weights_only=True);descs[k]=item['original_descriptor']
  def restore(k):
   d=descs[k];sz=d['storage_offset']+sum((n-1)*s for n,s in zip(d['shape'],d['stride']))+1
   buf=torch.empty(sz,device='cuda',dtype=saved[k].dtype);v=torch.as_strided(buf,d['shape'],d['stride'],d['storage_offset']);v.copy_(saved[k]);return v
  x,z,w=(restore(k) for k in ('x','z','weight'));r['gpu_executed']=True
  assert tuple(z.stride())==(12288,128,1) and z.storage_offset()==8192
  def digest(t):return hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  before=[digest(t) for t in (x,z,w)]
  def persist(k,t):
   p=a.output/(k+'.pt');torch.save(t.detach().cpu().contiguous(),p);r['artifacts'][k]=dict(path=str(p),sha256=sha(p))
  with torch.inference_mode():
   # Match original public wrapper reshape. Noncontiguous Z reshape may allocate;
   # record actual stride/storage so that this is observable, not assumed.
   xf=x.reshape(-1,128);zf=z.reshape(-1,128)
   r['helper_specialization']=dict(M=xf.shape[0],N=128,x_stride=list(xf.stride()),z_stride=list(zf.stride()),z_reshape_storage_same=zf.untyped_storage().data_ptr()==z.untyped_storage().data_ptr(),ROWS_PER_BLOCK=calc_rows_per_block(xf.shape[0],x.device),BLOCK_N=128,num_warps=1,eps=1e-6,norm_before_gate=True,is_rms_norm=True,activation='silu',extra_compiledkernel_replay_launches=0)
   native,mean,nr=layer_norm_fwd(xf,w.contiguous(),None,1e-6,z=zf,group_size=None,norm_before_gate=True,is_rms_norm=True,activation='silu');assert mean is None
   clean=gated_rms_norm(x,z,w,1e-6)
   y=torch.empty_like(x);st=torch.empty((1696*32,2),device=x.device,dtype=torch.float32)
   ct.launch(torch.cuda.current_stream(x.device),(1696*32,1,1),probe,(x,z,w,y,st,1e-6));torch.cuda.synchronize()
   native=native.reshape(x.shape)
   r['helper_vs_saved_native']=metrics(native,saved['nativeY0']);r['clean_vs_saved_tile']=metrics(clean,saved['tileY0']);r['instrumented_vs_clean']=metrics(y,clean)
   r['clean_vs_native']=metrics(clean,native);r['inputs_preserved']=before==[digest(t) for t in (x,z,w)]
   nrc=nr.cpu().contiguous();sc=st.cpu();cr=sc[:,1].contiguous();finite=torch.isfinite(nrc)&torch.isfinite(cr)
   r['rstd']=dict(raw_mismatches=int((nrc.view(torch.int32)!=cr.view(torch.int32)).sum()),max_abs=float((nrc-cr).abs()[finite].max()),all_finite=bool(finite.all()))
   coords=(saved['nativeY0'].view(torch.int16)!=saved['tileY0'].view(torch.int16)).nonzero().tolist();r['error_rows']=[]
   for t,h,c in coords:
    i=t*32+h;r['error_rows'].append(dict(index=[t,h,c],native_rstd=float(nrc[i]),candidate_rstd=float(cr[i]),native_bits=int(nrc.view(torch.int32)[i]),candidate_bits=int(cr.view(torch.int32)[i]),variance=float(sc[i,0])))
   for k,t in [('native_output',native),('clean_output',clean),('instrumented_output',y),('native_rstd',nr),('candidate_stats',st)]:persist(k,t)
  r['public_triton_cache_files']=[dict(path=str(p),sha256=sha(p)) for p in (a.output/'triton').rglob('*') if p.is_file() and p.suffix in ('.json','.ttir','.ttgir','.llir','.ptx')]
  r['complete']=True;r['gate']=result_gate(r);verify_pins(R/'CPU-PINS.json');save();return 0 if r['gate']['raw_pass'] else 2
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
