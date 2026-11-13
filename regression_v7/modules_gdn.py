"""Original v7 decode v4, saved native data only; compact state scope."""
import json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parents[1]
BANKS=P/'gdn_decode_output_order_qualification_v1/BANKS.json'
BANKS_SHA='df78dd47c6417b23d8af66e22b6cdcbeb8fc0f025987a02c08290c399e5faac3'
def cpu_tests():return ['gdn_decode_core_qualification_v4.test_cpu','gdn_prefill_output_copy_lean_v2.test_cpu','gdn_prefill_output_copy_lean_v2.test_lifetime','regression_v7.test_gdn']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dependencies():
 assert sha(BANKS)==BANKS_SHA,'Actual-bank manifest changed'
 result={str(BANKS):BANKS_SHA}
 for b in json.loads(BANKS.read_text()):
  for item in [*b['inputs'].values(),b['before'],b['native_state'],b['native_output']]:result[item['path']]=item['sha256']
 return result
def exact_bytes(a,b):
 if a!=b:raise AssertionError('Raw output/state mismatch')
def assert_corruption_detected(raw_reference):
 assert raw_reference
 altered=bytes([raw_reference[0]^1])+raw_reference[1:]
 try:exact_bytes(raw_reference,altered)
 except AssertionError:return True
 raise AssertionError('Raw comparator accepted injected corruption')
EXPECTED_IDENTITIES=[('layer0', 'native'), ('layer0', 'v7'), ('layer8', 'native'), ('layer8', 'v7')]
def audit(r):
 try:
  rows=r['records'];assert r['complete']is True
  assert [(x['bank'],x['backend'])for x in rows]==EXPECTED_IDENTITIES
  fields=('passed','contract','output_raw','state_raw','finite','output_corruption_detected','state_corruption_detected','same_stream','nonalias')
  assert all(all(x[k]is True for k in fields)for x in rows)
  return True
 except (KeyError,TypeError,AssertionError):return False
def run_gpu(torch,output):
 for p,h in dependencies().items():assert sha(p)==h,'Missing or altered external actual fixture'
 from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode as native
 from tileir_gdn_decode_core_v4 import packed_decode
 output=Path(output);output.mkdir(parents=True,exist_ok=True);r=dict(complete=False,records=[],scope='two actual banks, compact remapped 33-slot state; unselected original owner not reconstructed; input readonly is logical bytes only, strided allocation padding unproven')
 def save():(output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 def load(x):return torch.load(x['path'],weights_only=True,map_location='cpu')
 def raw(x):return x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
 try:
  for b in json.loads(BANKS.read_text()):
   cpu={k:load(v)for k,v in b['inputs'].items()};before=load(b['before']);refstate=load(b['native_state']);refout=load(b['native_output'])
   lookup={s:i+1 for i,s in enumerate(b['selected_slots'])};mapped=[lookup[s]for s in cpu['ssm_state_indices'].tolist()];assert len(lookup)==len(set(mapped))==32
   for label,fn in [('native',native),('v7',packed_decode)]:
    inp={}
    for k,t in cpu.items():
     x=torch.empty_strided(t.shape,b['inputs'][k]['metadata']['stride'],dtype=t.dtype,device='cuda');x.copy_(t);inp[k]=x
    inp['ssm_state_indices'].copy_(torch.tensor(mapped,dtype=torch.int32,device='cuda'));snap={k:raw(x)for k,x in inp.items()}
    state=torch.full((33,32,128,128),-7.,dtype=torch.float32,device='cuda');state[1:].copy_(before);sentinel=raw(state[0]);out=torch.empty(refout.shape,dtype=refout.dtype,device='cuda')
    tensors=[out,state,*inp.values()];ranges=[(t.untyped_storage().data_ptr(),t.untyped_storage().data_ptr()+t.untyped_storage().nbytes())for t in tensors]
    nonalias=all(a[1]<=b[0]or b[1]<=a[0]for i,a in enumerate(ranges)for b in ranges[i+1:]);assert nonalias
    stream=torch.cuda.current_stream(out.device).cuda_stream
    got=fn(**inp,scale=128**-0.5,initial_state=state,out=out,use_qk_l2norm_in_kernel=True);torch.cuda.synchronize()
    row=dict(bank=b['name'],backend=label,nonalias=nonalias,same_stream=torch.cuda.current_stream(out.device).cuda_stream==stream,passed=False,contract=False,output_raw=False,state_raw=False);r['records'].append(row);save()
    output_bytes=raw(out);expected_output=raw(refout);exact_bytes(output_bytes,expected_output);row['output_raw']=True
    row['output_corruption_detected']=assert_corruption_detected(expected_output[:64])
    state_bytes=raw(state[1:]);expected_state=raw(refstate);exact_bytes(state_bytes,expected_state);row['state_raw']=True
    row['state_corruption_detected']=assert_corruption_detected(expected_state[:64])
    assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(state).all())
    row['finite']=True
    assert isinstance(got,tuple)and got[0]is out and got[1]is state and raw(state[0])==sentinel and all(raw(inp[k])==snap[k]for k in inp)
    assert row['same_stream']
    row['contract']=True;row['passed']=True;save()
    del got,inp,state,out,snap,output_bytes,expected_output,state_bytes,expected_state
  r['complete']=True;assert audit(r);save();return r
 except BaseException:r['complete']=False;save();raise
