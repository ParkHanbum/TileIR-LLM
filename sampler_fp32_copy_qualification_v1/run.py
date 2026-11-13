"""Native2 then TileIR2 on each actual immediate sampler cast bank. No timing."""
import argparse,json,os,signal,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path[:0]=[str(P),str(R)]
from packed_grouped_qualification_v1.contract import sha,verify_pins
def check(m):
 assert m['complete'] and m['observers_removed'] and not m['failed'] and len(m['rows'])==16
 assert [s['call_index'] for s in m['samples']]==[0,15]
 for s in m['samples']:
  for key,dt,nb in [('input','torch.bfloat16',15892480),('immediate_output','torch.float32',31784960)]:
   d=s[key]['metadata'];assert d['shape']==[32,248320] and d['stride']==[248320,1] and d['dtype']==dt and d['device']=='cuda:0' and d['storage_offset']==0 and d['storage_nbytes']==nb
  o=s['observations'];assert o['fresh'] and o['cpu_cast_raw_equal'] and not any(o['input_nonfinite'].values()) and o['input_preserved_after_processors']
 return True
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha
 m=json.loads(Path(a.manifest).read_text());check(m)
 if not a.run_gpu:print(json.dumps(dict(executed=False,manifest_valid=True)));return 0
 os.fstat(int(os.environ['ALL_GEMM_LOCK_FD']));out=Path(a.output);out.mkdir(exist_ok=False)
 for env,name in [('CUDA_TILE_CACHE_DIR','tile'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  path=out/name;path.mkdir();os.environ[env]=str(path.resolve())
 result=dict(complete=False,gpu_executed=True,manifest_sha=a.manifest_sha,records=[],cache_environment={k:os.environ[k] for k in ('CUDA_TILE_CACHE_DIR','CUDA_TILE_TEMP_DIR','CUDA_CACHE_PATH','TMPDIR')},scope='Two actual saved activation banks; exact-sized owners restored, no timing/fullmodel support claim')
 def save():(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('175s worker')));signal.alarm(175)
 try:
  import torch
  from tileir_vllm_sampler_fp32_copy_v1 import sampler_fp32_copy
  def digest(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
  def load(e):
   assert sha(e['path'])==e['sha256'];t=torch.load(e['path'],map_location='cpu',weights_only=True)
   assert digest(t)==e['logical_sha256'] and list(t.shape)==e['metadata']['shape'] and str(t.dtype)==e['metadata']['dtype'];return t
  def raw(a,b):return torch.equal(a.view(torch.uint8),b.view(torch.uint8))
  with torch.inference_mode():
   for s in m['samples']:
    x,expected=[load(e).cuda() for e in [s['input'],s['immediate_output']]]
    phase='first' if s['call_index']==0 else 'last'
    assert x.storage_offset()==0 and x.stride()==(248320,1) and x.untyped_storage().nbytes()==15892480
    before=digest(x);outputs=[];checks=[]
    for kind,fn in [('native',lambda:torch.empty_like(x,dtype=torch.float32).copy_(x))]*2+[('tile',lambda:sampler_fp32_copy(x))]*2:
     y=fn();torch.cuda.synchronize()
     assert y.shape==expected.shape and y.dtype==torch.float32 and y.stride()==(248320,1) and y.storage_offset()==0 and y.untyped_storage().nbytes()==31784960
     fresh=all(y.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr() for t in [x,expected]+outputs)
     eq=raw(y,expected);finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all() and torch.isfinite(expected).all());strict=bool(torch.allclose(y.float(),expected.float(),atol=1e-5,rtol=1e-5));unchanged=before==digest(x)
     checks.append(dict(kind=kind,raw_equal=eq,finite=finite,strict=strict,fresh=fresh,input_unchanged=unchanged));outputs.append(y)
     if not all((eq,finite,strict,fresh,unchanged)):
      torch.save(dict(output=y.cpu(),expected=expected.cpu()),out/'first-failure.pt');result['records'].append(dict(phase=phase,calls=checks,passed=False));save();return 2
    result['records'].append(dict(phase=phase,calls=checks,raw_equal=True,finite=True,input_unchanged=True,reference_repeat_raw=raw(outputs[0],outputs[1]),candidate_repeat_raw=raw(outputs[2],outputs[3]),passed=True));save()
    del outputs,x,expected,y
  result['complete']=True;return 0
 except BaseException as e:result['error']=repr(e);raise
 finally:signal.alarm(0);save()
if __name__=='__main__':raise SystemExit(main())
