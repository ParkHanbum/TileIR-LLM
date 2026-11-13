"""Default CPU plan only. --execute-gpu requires a separately coordinated window."""
import argparse,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent.parent
MANIFEST=P/'packed_runtime_contract_capture_v1/packedCapture01/bank/MANIFEST.json'
EXPECTED='719a7fb82ab3f9be0a9ee8383226a47811bf5ba8686542347a0c3b6e35bd6a08'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-gpu',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
 assert sha(MANIFEST)==EXPECTED,'manifest changed'
 manifest=json.loads(MANIFEST.read_text()); samples=[s for s in manifest['samples'] if s['kind']=='gate_up']
 assert {s['phase'] for s in samples}=={'prefill','decode'} and len(samples)==2
 plan={'gpu_executed':False,'samples':[{'phase':s['phase'],'path':s['y']['path'],'shape':s['y']['metadata']['shape']} for s in samples], 'reference':'torch.ops._C.silu_and_mul(out,x), NOT Inductor fused graph kernel','checks':['original op repeat raw','input preservation','supplied output return/shape','raw equality','finite/mismatch/max absolute difference'],'timing':False}
 if not a.execute_gpu:print(json.dumps(plan,indent=2));return
 if a.output is None:raise ValueError('--output required for execution')
 # No torch/vllm imports occur in default plan mode.
 import torch
 import vllm._custom_ops  # registers installed public custom operators
 from tileir_vllm_silu_mul_v2 import silu_and_mul
 records=[]
 for s in samples:
  f=Path(s['y']['path']);assert sha(f)==s['y']['sha256']
  cpu=torch.load(f,map_location='cpu',weights_only=True)
  assert isinstance(cpu,torch.Tensor),'unexpected stored format'
  x=cpu.cuda();before=x.clone();shape=(x.shape[0],9216)
  r=torch.full(shape,float('nan'),dtype=x.dtype,device=x.device);repeat=torch.empty_like(r);actual=torch.full_like(r,float('nan'))
  returned=torch.ops._C.silu_and_mul(r,x);torch.ops._C.silu_and_mul(repeat,x)
  got=silu_and_mul(actual,x);torch.cuda.synchronize()
  raw=lambda t:t.contiguous().view(torch.int16)
  unchanged=torch.equal(raw(x),raw(before));repeat_ok=torch.equal(raw(r),raw(repeat))
  assert returned is None and got is None and unchanged and repeat_ok
  finite=bool(torch.isfinite(r).all() and torch.isfinite(actual).all())
  record={'phase':s['phase'],'shape':list(x.shape),'input_sha256':s['y']['sha256'],'reference_repeat_raw':repeat_ok,'input_unchanged':unchanged,'finite':finite,'raw_equal':torch.equal(raw(r),raw(actual)),'raw_mismatch_count':int((raw(r)!=raw(actual)).sum()),'max_abs_error':float((r.float()-actual.float()).abs().max()),'tolerance_verdict':'not set; raw and errors reported separately'}
  records.append(record)
  del x,before,r,repeat,actual
 with a.output.open('x') as f:json.dump({'gpu_executed':True,'records':records,'timing':False},f,indent=2)
 print(json.dumps(records,indent=2))
if __name__=='__main__':main()
