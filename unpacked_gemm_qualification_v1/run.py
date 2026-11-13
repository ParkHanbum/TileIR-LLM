"""Existing TileIRBLAS vs native D99 on 8 actual banks; no model or timing."""
import argparse,hashlib,json,os,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from packed_contiguous_split2_qualification_v1.run import compare,direct_default99
from contract import validate,passing,schedule,result_gate,CHUNK_BYTES,FAILURE_CAP

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',type=Path);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 assert sha(a.manifest)==a.manifest_sha;verify_pins(R/'CPU-PINS.json');m=json.loads(Path(a.manifest).read_text());groups=validate(m)
 if not a.run_gpu:print(json.dumps(dict(executed=False,contracts=5,banks=8,maximum_calls=32,timing=False,model_replacement=False)));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.fstat(fd);target=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').stat();assert (st.st_dev,st.st_ino)==(target.st_dev,target.st_ino)
 assert a.output and a.output.is_absolute();a.output.mkdir(exist_ok=False)
 for key,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton')]:
  (a.output/name).mkdir();os.environ[key]=str(a.output/name)
 r=dict(finished=False,complete=False,gpu_executed=False,adopted=False,timing=False,manifest_sha=a.manifest_sha,contracts=[],logical_calls=0,failure_bytes=0,reference_limitations='Fresh independent HOST handle/workspace; captured pointer mode unknown. Fixed D99/math0/compute68/native scalar1/0; each bank must reproduce captured Y before TileIR. No performance or full-model qualification.')
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('570s worker bound')));signal.alarm(570)
 torch=None;ref=None
 try:
  import torch
  from safetensors import safe_open
  from tileirblas_once_occ2 import linear
  from tileirblas_once_occ2.api import selected_path
  torch.set_num_threads(4)
  flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
  r['initial_flags']=flags();assert r['initial_flags']==[True,True]
  ref,call,libpath=direct_default99(torch);r.update(gpu_executed=True,reference_version=ref.version,reference_library=dict(path=str(libpath),sha256=sha(libpath)));save()
  torch.cuda.reset_peak_memory_stats()
  def digest(t):
   assert t.is_contiguous();flat=t.detach().view(-1);h=hashlib.sha256()
   for start in range(0,flat.numel(),CHUNK_BYTES//t.element_size()):h.update(flat[start:start+CHUNK_BYTES//t.element_size()].cpu().view(torch.uint8).numpy().tobytes())
   return h.hexdigest()
  def execute(row,bank):
   s=row['sample'];weight=row['weight'];source=weight['source'];bank['source']=dict(module=s['module'],X=s['X']['path'],Y=s['Y']['path'],weight=source)
   cpu={}
   for key in ('X','Y'):
    v=s[key];assert sha(v['path'])==v['sha256'];t=torch.load(v['path'],map_location='cpu',weights_only=True);assert t.dtype==torch.bfloat16 and list(t.shape)==v['metadata']['shape'] and list(t.stride())==v['saved_stride'];assert digest(t)==v['logical_sha256'] and bool(torch.isfinite(t).all());cpu[key]=t
   meta=weight['runtime_metadata'];N,K=meta['shape'];M=cpu['X'].shape[0]
   q=Path(source['path']);st=q.stat();assert str(q.resolve())==source['resolved_path'] and (st.st_size,st.st_mtime_ns)==(source['size'],source['mtime_ns'])
   w=torch.empty((N,K),dtype=torch.bfloat16,device='cuda');h=hashlib.sha256();step=max(1,CHUNK_BYTES//(K*2))
   with safe_open(str(q),framework='pt',device='cpu') as f:
    view=f.get_slice(source['key']);assert view.get_shape()==[N,K]
    for start in range(0,N,step):
     chunk=view[start:min(start+step,N),:].contiguous();assert chunk.dtype==torch.bfloat16 and bool(torch.isfinite(chunk).all());h.update(chunk.view(torch.uint8).numpy().tobytes());w[start:min(start+step,N)].copy_(chunk)
   del chunk
   assert h.hexdigest()==weight['checkpoint_named_tensor_sha256']==weight['logical_sha256']
   x=cpu['X'].to('cuda');y=torch.empty((M,N),dtype=torch.bfloat16,device='cuda')
   assert tuple(w.stride())==tuple(meta['stride']) and w.storage_offset()==0 and w.untyped_storage().nbytes()==meta['storage_nbytes']
   assert list(x.stride())==s['X']['metadata']['stride'] and x.storage_offset()==0 and x.untyped_storage().nbytes()==s['X']['metadata']['storage_nbytes']
   before=(digest(x),digest(w));assert before==(s['X']['logical_sha256'],weight['logical_sha256']);bank['weight_checkpoint_equal']=True
   bank['backing_scope']='Exact captured dense storage spans, zero offset and no holes; full X/W storage bytes hashed after each call.'
   bank['selected_tileir_path']=selected_path(M,N,K)
   def preserved():
    assert flags()==[True,True]
    assert (q.stat().st_size,q.stat().st_mtime_ns)==(source['size'],source['mtime_ns'])
    ok=(digest(x),digest(w))==before
    assert ok,'Input/backing mutation: abort entire run';return ok
   for stage in ('native','tileir'):
    for repeat in range(2):
     rec=dict(stage=stage,repeat=repeat);bank['records'].append(rec);save();y.fill_(float('nan'))
     if stage=='native':rec['api']=call(x,w,y);output=y
     else:output=linear(x,w,out=y)
     r['logical_calls']+=1;torch.cuda.synchronize()
     rec['output_contract_pass']=(output is y and list(y.shape)==s['Y']['metadata']['shape'] and list(y.stride())==s['Y']['metadata']['stride'] and y.dtype==torch.bfloat16 and y.device==x.device and y.storage_offset()==0 and y.untyped_storage().nbytes()==s['Y']['metadata']['storage_nbytes'] and y.untyped_storage().data_ptr() not in (x.untyped_storage().data_ptr(),w.untyped_storage().data_ptr()))
     assert rec['output_contract_pass'],'Output shape/storage/alias contract failure'
     out=y.cpu();rec['comparison']=compare(cpu['Y'],out,torch);rec['output_sha256']=digest(out);rec['input_unchanged']=rec['backing_unchanged']=preserved();save()
     if not passing(rec['comparison']):
      bank.update(state='HOLD',inputs_unchanged=True,backing_unchanged=True)
      size=out.numel()*out.element_size();assert r['failure_bytes']+size<=FAILURE_CAP
      path=a.output/(row['label']+'-'+stage+'-failure.pt');torch.save(out,path);rec['failure_output']=dict(path=str(path),sha256=sha(path));r['failure_bytes']+=size;save();return
   bank.update(state='PASS',inputs_unchanged=preserved(),backing_unchanged=True);save()
  schedule(groups,execute,r['contracts'],save)
  r['peak_allocated_bytes']=torch.cuda.max_memory_allocated();r['peak_reserved_bytes']=torch.cuda.max_memory_reserved();r['finished']=True
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  errors=[]
  def cleanup(stage,fn):
   try:fn()
   except BaseException as e:errors.append(dict(stage=stage,error=repr(e)))
  if torch is not None:
   cleanup('synchronize',lambda:torch.cuda.synchronize())
   def final_flags():
    r['final_flags']=[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k];assert r['final_flags']==r.get('initial_flags')==[True,True]
   cleanup('readonly_flags',final_flags)
  if ref is not None:
   def close():ref.close();r['reference_closed']=True
   cleanup('reference_close',close)
  def pins():verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha;r['source_pins_unchanged']=True
  cleanup('pins',pins);signal.alarm(0)
  if errors:r['cleanup_errors']=errors
  gate=result_gate(r);r.update(result_gate=gate,complete=gate['complete']);save()
 return 0 if gate['raw_pass'] else (2 if gate['raw_difference_hold'] else 1)
if __name__=='__main__':raise SystemExit(main())
