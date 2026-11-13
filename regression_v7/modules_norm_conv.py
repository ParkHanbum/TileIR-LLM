"""Draft original-v7 module regressions; no performance or full-family claim.
Reuses norm_three_stage_qualification_v1 (two real prefill boundaries) and
indexed_decode_conv_qualification_v1 (one real decode, original strides).
No gated norm, norm decode, prefill conv or post-conv coverage is claimed.
"""
import hashlib
import inspect
import json
from pathlib import Path
P = Path('/home/n/work/TileIR/public-tileir-blas-20261111')
NORM = (
 ('norm_no_residual', 'norm_runtime_integration_v3/normIntegration03/shadow/shadow-PROGRESS.json', '204c2581991aa51e6c1e647d0d71354cf25d132f08c43ba9860842cedc35cba0'),
 ('norm_with_residual', 'norm_runtime_integration_v2/normIntegration02/shadow/shadow-PROGRESS.json', 'baea2b56dd948f41a7b376a278106e035c4fd8a2b54c777235fa81b46ecf2e65'))
CONV = P/'indexed_conv_runtime_capture_v2/indexedConvCapture01/bank/MANIFEST.json'
CONV_SHA = '18bc94fb1c3252fe3a66326386265f35ad4f3372ef7fcd9016fd8b7ea2f77c35'
EXPECTED = [(n,b) for n in ('norm_no_residual','norm_with_residual','conv_decode') for b in ('native','v7')]
SOURCE_PINS = {'/home/n/work/TileIR/public-tileir-blas-20261111/tileir_vllm_gemma_norm_v3/api.py': '96e83e86867fc7492dfcc63c46da41c775ecc74616e1eaebe60cb590d29cc719', '/home/n/work/TileIR/public-tileir-blas-20261111/tileir_vllm_gemma_norm_v3/kernels.py': 'e03b39da5b9481aba219f01cabc0dcf11018d058de9083e929bfdf076819056d', '/home/n/work/TileIR/public-tileir-blas-20261111/tileir_indexed_decode_conv_v1/api.py': 'dc35c9c93ef3ab63b330d3f6af2933825fa5c22050bb77b7060b0cc1e0800d21', '/home/n/work/TileIR/public-tileir-blas-20261111/tileir_indexed_decode_conv_v1/kernels.py': '1f106f645703c5c6216f77f37b342fdecfd6006e80386e68eb114380a6f760c4', '/home/n/work/TileIR/public-tileir-blas-20261111/norm_three_stage_qualification_v1/run.py': '78174bec6b3295f25d65a6688d873a1a2d5aee21978012e3c72a564ce701b4ec', '/home/n/work/TileIR/public-tileir-blas-20261111/indexed_decode_conv_qualification_v1/run.py': '1f3d7513762a0aa4f1188215453673e52859d02714e399e61d8ff20e9328708b', '/home/n/work/TileIR/public-tileir-blas-20261111/norm_runtime_qualification_v2/compare.py': '5006a0049224e269d03c172011f8be3d87b4651d122b471c278b2f9c920b6a0c', '/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/vllm/ir/ops/layernorm.py': 'b7621f663a97caf6908507b6cbe03adf23d0001a441d44af2b24c1ad0fc8bf57', '/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/vllm/model_executor/layers/mamba/ops/causal_conv1d.py': 'cb16cc9250c4195c09d6d83b43bba607c2d749b621d8d135a920443d1577265f'}


def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def manifests():
 norms=[]
 for name,path,digest in NORM:
  path=P/path
  if sha(path)!=digest:raise AssertionError('Norm manifest changed')
  first=json.loads(path.read_text())['first_mismatch']
  required={'x','weight','nativeY0'} | ({'residual','nativeY1'} if name=='norm_with_residual' else set())
  assert required<=set(first['artifacts'])
  norms.append((name,{k:v for k,v in first['artifacts'].items() if k in required}))
 assert sha(CONV)==CONV_SHA
 m=json.loads(CONV.read_text());assert m['complete'] and m['schema']=='indexed-causal-conv-native-v1'
 conv,=[s for s in m['samples'] if s['phase']=='decode']
 assert conv['original_calls']==1 and conv['state_unselected_unchanged'] and conv['weight_unchanged'] and not conv['skip_positions']
 assert conv['arguments']['out'] is None and conv['arguments']['bias'] is None and conv['arguments']['activation']=='silu'
 assert conv['state_descriptor']['shape']==[311,8192,3] and conv['state_descriptor']['stride']==[1081344,1,8192]
 assert conv['slots']==list(range(4,129,4))
 return norms,conv

def dependencies():
 norms,c=manifests();d={str(P/path):h for _,path,h in NORM};d[str(CONV)]=CONV_SHA;d.update(SOURCE_PINS)
 for _,items in norms:
  for item in items.values():d[item['path']]=item['sha256']
 for item in [*c['pre'].values(),c['selected_state_before'],c['selected_state_after'],c['post_x'],c['output']]:
  d[item['path']]=item['sha256']
 return d

def cpu_tests():return ['tileir_vllm_gemma_norm_v3.test_cpu','tileir_indexed_decode_conv_v1.test_cpu','regression_v7.test_norm_conv']

def span(m):
 assert len(m['shape'])==len(m['stride']) and all(type(n)is int and n>0 for n in m['shape']+m['stride'])
 assert type(m['storage_offset'])is int and m['storage_offset']>=0
 return m['storage_offset']+1+sum((n-1)*s for n,s in zip(m['shape'],m['stride']))

def require_metric(m):
 assert m['raw_equal']is True and m['strict_pass']is True
 assert all(type(m[k])is int and m[k]==0 for k in ('raw_mismatches','strict_failures','candidate_nonfinite','reference_nonfinite'))

def assert_corruption_detected(value,torch,compare=None):
 """Exercise the SAME numeric gate on private CPU output/state samples.
 Copy at most32 words; never modify input value or original GPU storage.
 """
 assert isinstance(value,torch.Tensor) and value.device.type=='cpu'
 assert value.dtype==torch.bfloat16 and value.numel()>0
 if compare is None:
  from norm_runtime_qualification_v2.compare import metrics as compare
 reference=value.detach().contiguous().reshape(-1)[:32].clone()
 assert bool(torch.isfinite(reference).all())
 original=reference.view(torch.int16).clone()
 # Positive control prevents an always-reject comparator passing this test.
 require_metric(compare(reference,reference))
 for mode in ('bit','nan'):
  bad=reference.clone();words=bad.view(torch.int16)
  if mode=='bit':words[0]^=1
  else:words[0]=0x7fc1
  verdict=compare(bad,reference)
  try:require_metric(verdict)
  except AssertionError:pass
  else:raise AssertionError('Numeric detector accepted deliberate '+mode+' corruption')
 assert torch.equal(reference.view(torch.int16),original)
 return True

def audit(r):
 try:
  assert r['complete']is True and r['context_restored']is True and r['timing']is False
  assert not r.get('error') and not r.get('cleanup_error') and r['scope']=='two prefill norm boundaries and one decode convolution; not complete families'
  assert [(x['case'],x['backend'])for x in r['records']]==EXPECTED
  for x in r['records']:
   assert all(x[k]is True for k in ('contract','same_stream','input_contract','corruption_detected'))
   expected=1 if x['case']=='norm_no_residual' else 2
   assert len(x['metrics'])==expected
   counts=[4341760] if x['case']=='norm_no_residual' else [4341760,4341760] if x['case']=='norm_with_residual' else [262144,786432]
   for m,n in zip(x['metrics'],counts):require_metric(m);assert m['elements']==n
   if x['case'].startswith('norm_'):assert x['fresh_outputs']is True and x['full_input_backing_readonly']is True
   else:assert all(x[k]is True for k in ('selected_state_exact','state_minspan_exact','x_minspan_exact','weight_indices_readonly','distinct_same_storage_return'))
  return dict(passed=True,records=6,norm_boundaries=2,conv_boundaries=1,full_family_coverage=False)
 except (AssertionError,KeyError,TypeError,ValueError):return dict(passed=False)


def prepare_output(output):
 output=Path(output)
 assert output.is_absolute() and output==output.resolve()
 assert not output.is_relative_to(P) and not P.is_relative_to(output)
 output.mkdir(parents=True,exist_ok=False)
 return output

def save_partial(output,result):
 with (output/'RESULT.json').open('x') as f:
  json.dump(result,f,indent=2);f.write('\n')

def run_gpu(torch,output):
 for p,h in dependencies().items():assert sha(p)==h,('External dependency changed',p)
 from vllm import ir
 from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
 from tileir_vllm_gemma_norm_v3 import gemma_rms_norm
 from tileir_indexed_decode_conv_v1 import causal_conv1d_update_indexed
 from norm_runtime_qualification_v2.compare import metrics
 norms,c=manifests()
 native_norms=(ir.ops.rms_norm.impls['native'].impl_fn,ir.ops.fused_add_rms_norm.impls['native'].impl_fn)
 for fn in (*native_norms,causal_conv1d_update):
  path=inspect.getsourcefile(fn);assert path in SOURCE_PINS and sha(path)==SOURCE_PINS[path]
 def raw(t):return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
 def digest(t):return hashlib.sha256(raw(t)).hexdigest()
 def load(item):
  t=torch.load(item['path'],map_location='cpu',weights_only=True)
  if 'logical_sha256'in item:assert digest(t)==item['logical_sha256']
  return t
 def restore(v,md):
  assert list(v.shape)==md['shape'] and str(v.dtype)==md['dtype']
  host=torch.full((span(md),),-7,dtype=v.dtype)
  torch.as_strided(host,md['shape'],md['stride'],md['storage_offset']).copy_(v)
  backing=host.to(device=device)
  return backing,torch.as_strided(backing,md['shape'],md['stride'],md['storage_offset']),host
 device=torch.cuda.current_device();prior=torch.cuda.current_stream(device);stream=torch.cuda.Stream(device=device)
 r=dict(complete=False,timing=False,records=[],scope='two prefill norm boundaries and one decode convolution; not complete families')
 output=prepare_output(output)
 held=[]
 try:
  with torch.cuda.device(device),torch.cuda.stream(stream),torch.inference_mode():
   for name,items in norms:
    cpu={k:load(v)for k,v in items.items()};has_res='residual'in cpu
    assert tuple(cpu['x'].shape)==(1696,2560) and cpu['x'].dtype==torch.bfloat16
    for label in ('native','v7'):
     inp={};backs=[];snap=[]
     for k in ('x','weight')+(('residual',)if has_res else ()):
      backing,value,host=restore(cpu[k],items[k]['original_descriptor']);backs.append(backing);snap.append(digest(host));inp[k]=value
     held[:]=backs;res=inp.get('residual');x,w=inp['x'],inp['weight']
     if label=='native':out=native_norms[1](x,res,w.float()+1.,1e-6) if has_res else native_norms[0](x,w.float()+1.,1e-6)
     else:out=gemma_rms_norm(x,w,1e-6,res)
     assert isinstance(out,tuple)==has_res
     outputs=list(out)if has_res else[out];held.extend(outputs)
     assert len(outputs)==(2 if has_res else 1)
     stream.synchronize();checks=[]
     for i,t in enumerate(outputs):
      assert t.dtype==torch.bfloat16 and tuple(t.shape)==(1696,2560) and t.is_contiguous() and t.storage_offset()==0
      checks.append(metrics(t,cpu['nativeY'+str(i)]));require_metric(checks[-1])
     ptrs=[t.untyped_storage().data_ptr()for t in backs+outputs];assert len(ptrs)==len(set(ptrs))
     assert [digest(t)for t in backs]==snap
     same=torch.cuda.current_stream(device)==stream;assert same
     r['records'].append(dict(case=name,backend=label,metrics=checks,contract=True,input_contract=True,same_stream=same,fresh_outputs=True,full_input_backing_readonly=True,corruption_detected=assert_corruption_detected(outputs[0].detach().cpu(),torch)))
     stream.synchronize();held.clear()
   vals={k:load(v)for k,v in c['pre'].items()};before=load(c['selected_state_before']);after=load(c['selected_state_after']);expected=load(c['output']);assert digest(expected)==digest(load(c['post_x']))
   sm=c['state_descriptor'];slots=c['slots'];assert span(sm)*2<=700*1024*1024
   for label in ('native','v7'):
    state_host=torch.full((span(sm),),-7.,dtype=torch.bfloat16);sv=torch.as_strided(state_host,sm['shape'],sm['stride'],sm['storage_offset']);sv[slots]=before
    state_back=state_host.to(device=device);state=torch.as_strided(state_back,sm['shape'],sm['stride'],sm['storage_offset'])
    xb,x,xhost=restore(vals['x'],c['pre']['x']['metadata']);wb,w,whost=restore(vals['weight'],c['pre']['weight']['metadata']);ib,idx,ihost=restore(vals['conv_state_indices'],c['pre']['conv_state_indices']['metadata'])
    held[:]=[state_back,xb,wb,ib];sv[slots]=after
    torch.as_strided(xhost,c['pre']['x']['metadata']['shape'],c['pre']['x']['metadata']['stride'],c['pre']['x']['metadata']['storage_offset']).copy_(expected)
    if label=='native':out=causal_conv1d_update(x,state,w,None,'silu',conv_state_indices=idx,validate_data=False)
    else:out=causal_conv1d_update_indexed(x,state,w,idx)
    held.append(out);stream.synchronize()
    checks=[metrics(out,expected),metrics(state[slots],after)]
    for check in checks:require_metric(check)
    assert out is not x and out.untyped_storage().data_ptr()==x.untyped_storage().data_ptr() and out.stride()==x.stride() and out.storage_offset()==x.storage_offset()
    assert digest(state_back)==digest(state_host) and digest(xb)==digest(xhost)
    assert digest(wb)==digest(whost) and digest(ib)==digest(ihost)
    same=torch.cuda.current_stream(device)==stream;assert same
    r['records'].append(dict(case='conv_decode',backend=label,metrics=checks,contract=True,input_contract=True,same_stream=same,selected_state_exact=True,state_minspan_exact=True,x_minspan_exact=True,weight_indices_readonly=True,distinct_same_storage_return=True,corruption_detected=assert_corruption_detected(out.detach().cpu(),torch) and assert_corruption_detected(state[slots].detach().cpu(),torch)))
    stream.synchronize();held.clear()
 except BaseException as error:
  r['error']=repr(error);raise
 finally:
  try:
   stream.synchronize();held.clear()
   assert torch.cuda.current_device()==device and torch.cuda.current_stream(device)==prior
   r['context_restored']=True
   if 'error' not in r:
    r['complete']=True
    assert audit(r)['passed']
  except BaseException as error:
   r['complete']=False;r['cleanup_error']=repr(error);raise
  finally:save_partial(output,r)
 return r
