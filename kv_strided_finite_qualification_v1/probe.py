"""GPU body draft only. No CLI/automatic execution; coordinator lifecycle pending."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent.parent
MANIFEST=P/'attention_kv_runtime_capture_v1/attentionKVCapture01/bank/MANIFEST.json'
CASES=('actual','padding_nan','first_nan','last_nan','first_posinf','last_neginf','negative_zero','subnormal','maxfinite')
def run(save):
 # Called only by a future reviewed bounded coordinator harness.
 import torch
 from tileir_kv_strided_finite_v1.api import all_finite_value
 result=dict(complete=False,records=[],scope='Historical logical value bank replay with reconstructed owner sentinel, not original full-owner capture')
 m=json.loads(MANIFEST.read_text());assert m['complete'] and m['schema']=='attention-kv-capture-v1'
 for sample in m['samples']:
  item=sample['pre_inputs']['value'];meta=item['metadata'];path=Path(item['path'])
  assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
  data=torch.load(path,map_location='cpu',weights_only=True)
  assert tuple(data.shape)==tuple(meta['shape']) and str(data.dtype)==meta['dtype']
  n=meta['shape'][0];assert n in (32,1696)
  for case in CASES:
   result['active_case']=[sample['phase'],case];save(result)
   owner=torch.full((n*10240,),3.,dtype=torch.bfloat16,device='cuda')
   x=owner.as_strided((n,4,256),(10240,256,1),9216);x.copy_(data)
   expected=True
   if case=='padding_nan':owner[0]=float('nan');owner[9215]=float('nan');owner[10240]=float('nan')
   elif case=='first_nan':x[0,0,0]=float('nan');expected=False
   elif case=='last_nan':x[-1,-1,-1]=-float('nan');expected=False
   elif case=='first_posinf':x[0,0,0]=float('inf');expected=False
   elif case=='last_neginf':x[-1,-1,-1]=-float('inf');expected=False
   elif case=='negative_zero':x[-1,-1,-1]=-0.
   elif case=='subnormal':x[-1,-1,-1]=2.**-133
   elif case=='maxfinite':x[-1,-1,-1]=torch.finfo(torch.bfloat16).max
   before=owner.view(torch.uint8).cpu();desc=(x.data_ptr(),x.storage_offset(),x.stride(),x.is_neg(),x.is_conj());stream=torch.cuda.current_stream(x.device).cuda_stream
   for arm in ('native','tile'):
    for repeat in range(2):
     value=bool(torch.isfinite(x).all()) if arm=='native' else all_finite_value(x)
     preserved=torch.equal(before,owner.view(torch.uint8).cpu())
     assert desc==(x.data_ptr(),x.storage_offset(),x.stride(),x.is_neg(),x.is_conj())
     assert torch.cuda.current_stream(x.device).cuda_stream==stream
     row=dict(phase=sample['phase'],case=case,arm=arm,repeat=repeat,value=value,expected=expected,owner_preserved=preserved,passed=value is expected and preserved)
     result['records'].append(row);save(result);assert row['passed'],'Predicate or owner mismatch; stop without retry'
   del x,owner,before
  del data
 result['complete']=True;save(result);return result
