"""Original v7 leaf fixtures only. Coordinator owns admission, final sync and lifecycle.
QK backing is reconstructed from saved storage, including alias/padding. SiLU
uses actual gate_up inputs and fresh public native references, not invented goldens.
No whole-model, universal input, asynchronous lifetime or performance claim.
"""
import hashlib,json,math
from pathlib import Path
P=Path('/home/n/work/TileIR/public-tileir-blas-20261111')
MANIFESTS={
'qk_rope_runtime_capture_v1/qkRopeCapture01/bank/MANIFEST.json':'1953f2a5cc9cfae4f34197cb84e65a85180839a6d62c725cc995d4ce070e06df',
'packed_runtime_contract_capture_v1/packedCapture01/bank/MANIFEST.json':'719a7fb82ab3f9be0a9ee8383226a47811bf5ba8686542347a0c3b6e35bd6a08'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def banks():
 ms=[]
 for rel,h in MANIFESTS.items():
  assert sha(P/rel)==h;ms.append(json.loads((P/rel).read_text()))
 q=ms[0]['samples'];s=[x for x in ms[1]['samples'] if x['kind']=='gate_up']
 assert [x['phase'] for x in q]==['prefill','decode']
 s=sorted(s,key=lambda x:x['phase']!='prefill');assert [x['phase'] for x in s]==['prefill','decode']
 return q,s
def dependencies():
 q,s=banks();d={str(P/k):v for k,v in MANIFESTS.items()}
 for sample in q:
  for t in [*sample['pre'].values(),*sample['outputs']]:
   d[t['path']]=t['sha256'];d[t['storage_path']]=t['storage_sha256']
 for sample in s:d[sample['y']['path']]=sample['y']['sha256']
 return d
def cpu_tests():return ['regression_v7.test_pointwise_remaining']
def identities():return [(f,p,b,i) for f in ('qk_rope','silu') for p in ('prefill','decode') for b in ('native','v7') for i in (0,1)]
def shapes(f,p):
 n=1696 if p=='prefill' else 32
 return [[n,4096],[n,1024],[n,4096]] if f=='qk_rope' else [[n,9216]]
def descriptor(t):return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),offset=t.storage_offset(),elements=t.numel())
def expected(shape):return dict(shape=shape,stride=[shape[1],1],dtype='torch.bfloat16',offset=0,elements=math.prod(shape))
def compare_bytes(a,b):
 assert len(a)==len(b),'Byte extent mismatch'
 return dict(raw_equal=a==b,raw_mismatches=sum(x!=y for x,y in zip(a,b)))
def audit(r):
 try:
  assert r['complete'] is True and len(r['records'])==16
  for row,identity in zip(r['records'],identities()):
   assert tuple(row['identity'])==identity
   assert all(row[k] is True for k in ('readonly','alias','current_stream','return_contract','repeat_equal'))
   ss=shapes(*identity[:2]);assert len(row['outputs'])==len(ss)
   for m,s in zip(row['outputs'],ss):
    assert m['descriptor']==expected(s) and m['raw_equal'] is True
    for k in ('raw_mismatches','candidate_nonfinite','reference_nonfinite'):assert type(m[k]) is int and m[k]==0
  return True
 except (KeyError,TypeError,AssertionError):return False
def run_gpu(torch,output):
 for p,h in dependencies().items():assert sha(p)==h,'Fixture drift'
 from tileir_vllm_qk_rope_v1 import fused_qk_rmsnorm_rope_gate as tile_qk
 from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate as native_qk
 from tileir_vllm_silu_mul_v2 import silu_and_mul as tile_silu
 import vllm._custom_ops
 q,s=banks();output=Path(output);output.mkdir(parents=True,exist_ok=True)
 r=dict(complete=False,records=[],scope=__doc__,timing=False)
 def save():(output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 def raw(t):return t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
 def load(t):return torch.load(t['path'],map_location='cpu',weights_only=True)
 def record(identity,ys,refs,owners,before,stream,ret,prior):
  torch.cuda.synchronize();outbytes=[raw(y) for y in ys]
  storages=[y.untyped_storage().data_ptr() for y in ys]
  ms=[]
  for y,ref,bb in zip(ys,refs,outbytes):
   m=compare_bytes(bb,raw(ref));m.update(descriptor=descriptor(y),candidate_nonfinite=int((~torch.isfinite(y)).sum().item()),reference_nonfinite=int((~torch.isfinite(ref)).sum().item()));ms.append(m)
  row=dict(identity=identity,outputs=ms,readonly=before==[sha_bytes(raw(x)) for x in owners],alias=len(set(storages))==len(storages) and all(a!=x.untyped_storage().data_ptr() for a in storages for x in owners),current_stream=torch.cuda.current_stream().cuda_stream==stream,return_contract=ret,repeat_equal=prior is None or outbytes==prior)
  r['records'].append(row);save()
  assert all(row[k] is True for k in ('readonly','alias','current_stream','return_contract','repeat_equal'))
  for m,shape in zip(ms,shapes(*identity[:2])):
   assert m['descriptor']==expected(shape) and m['raw_equal'] and m['raw_mismatches']==m['candidate_nonfinite']==m['reference_nonfinite']==0,'First mismatch HOLD'
  return outbytes
 try:
  with torch.inference_mode():
   for sample in q:
    refs=[load(t) for t in sample['outputs']];prm=sample['parameters']
    for branch,fn in [('native',native_qk),('v7',tile_qk)]:
     prior=None
     for repeat in (0,1):
      buffers={};vals={}
      for name,t in sample['pre'].items():
       d=t['metadata'];key=(d['storage_ptr'],d['storage_bytes'])
       if key not in buffers:buffers[key]=torch.load(t['storage_path'],map_location='cpu',weights_only=True).cuda()
       vals[name]=torch.empty(0,dtype=getattr(torch,d['dtype'].split('.')[-1]),device='cuda').set_(buffers[key].untyped_storage(),d['storage_offset'],d['shape'],d['stride'])
      assert vals['q_gate'].untyped_storage().data_ptr()==vals['k'].untyped_storage().data_ptr()==vals['qkv'].untyped_storage().data_ptr()
      assert tuple(vals['positions'].stride())==(2049,1) and vals['k'].storage_offset()==8192
      owners=list(buffers.values());before=[sha_bytes(raw(x)) for x in owners];stream=torch.cuda.current_stream().cuda_stream
      ys=fn(*[vals[k] for k in ('q_gate','k','q_weight','k_weight','cache','positions')],prm['eps'],prm['num_q_heads'],prm['num_kv_heads'],prm['head_dim'],prm['rotary_dim'],mrope_section=prm['mrope_section'],norm_beta=prm['norm_beta'])
      assert isinstance(ys,tuple) and len(ys)==3
      prior=record(('qk_rope',sample['phase'],branch,repeat),ys,refs,owners,before,stream,True,prior)
   for sample in s:
    cpu=load(sample['y']);refs=None
    for branch,fn in [('native',torch.ops._C.silu_and_mul),('v7',tile_silu)]:
     prior=None
     for repeat in (0,1):
      x=cpu.cuda();y=torch.full((x.shape[0],9216),float('nan'),device='cuda',dtype=x.dtype)
      before=[sha_bytes(raw(x))];stream=torch.cuda.current_stream().cuda_stream;ret=fn(y,x);torch.cuda.synchronize()
      if refs is None:refs=[y.cpu()]
      prior=record(('silu',sample['phase'],branch,repeat),[y],refs,[x],before,stream,ret is None,prior)
  r['complete']=True;assert audit(r);save();return r
 except BaseException as e:r['complete']=False;r['error']=type(e).__name__+': '+str(e);save();raise
def sha_bytes(x):return hashlib.sha256(x).hexdigest()
