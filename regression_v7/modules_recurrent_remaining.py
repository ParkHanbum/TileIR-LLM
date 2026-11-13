"""Six saved actual boundaries, original-v7 APIs only; no full-family claim.
Default VllmConfig for captured gated public callable replay, not engine config.
Prefill conv uses original logical strides with minspan sentinel, NOT all bytes
of the original shared owner. No main extraction or numerical code alteration.
"""
import hashlib,json,inspect,math
from pathlib import Path
from types import SimpleNamespace
from regression_v7.modules_norm_conv import span,prepare_output,save_partial
P=Path('/home/n/work/TileIR/public-tileir-blas-20261111')
MANIFESTS={
 'gated':('gated_norm_runtime_capture_v2/gatedCapture01/bank/MANIFEST.json','56675e0286f6728c0dc967981d154e6531ac143929ddbf6c33abd474c38717fe'),
 'norm':('norm_runtime_contract_capture_v2/normCapture01/bank/MANIFEST.json','ed993c81d5ba67493a096635874b29efaf44e818001ff19de33ac62fbd9990a5'),
 'conv':('indexed_conv_runtime_capture_v2/indexedConvCapture01/bank/MANIFEST.json','18bc94fb1c3252fe3a66326386265f35ad4f3372ef7fcd9016fd8b7ea2f77c35'),
 'post':('gdn_post_conv_runtime_capture_v1/postConvCapture01/bank/MANIFEST.json','206f2219c13e0fb478daa52758407649eba1e54b27032293a12a9a1184485fec')}
CASES=('gated_prefill','gated_decode','norm_decode_no_residual','norm_decode_with_residual','conv_prefill','postconv_prefill')
COUNTS=(1,1,1,2,2,5)
SOURCE_PINS={'/home/n/work/TileIR/public-tileir-blas-20261111/tileir_vllm_gated_norm_ordered_v1/api.py': '41bccb58c6741086748939c918996df61393082e315154563685d2264c9faa9e', '/home/n/work/TileIR/public-tileir-blas-20261111/tileir_vllm_gated_norm_ordered_v1/kernels.py': 'd5d5ee094c3f729943593c325b4ccfa3e6d73f2001f8787e72ddc31166e154bb', '/home/n/work/TileIR/public-tileir-blas-20261111/tileir_vllm_gemma_norm_v3/api.py': '96e83e86867fc7492dfcc63c46da41c775ecc74616e1eaebe60cb590d29cc719', '/home/n/work/TileIR/public-tileir-blas-20261111/tileir_vllm_gemma_norm_v3/kernels.py': 'e03b39da5b9481aba219f01cabc0dcf11018d058de9083e929bfdf076819056d', '/home/n/work/TileIR/public-tileir-blas-20261111/tileir_indexed_prefill_conv_v1/api.py': '1a99ed6ca5c610b6bff7d31f1dcd073077ac25fc196ce721d1c98f1b7915f1cb', '/home/n/work/TileIR/public-tileir-blas-20261111/tileir_indexed_prefill_conv_v1/kernels.py': '60710b2d21c7cb54f7f3e0722bba2b663b1ac83539feedbb315fb975d5a6f679', '/home/n/work/TileIR/public-tileir-blas-20261111/tileir_vllm_post_conv_v2/api.py': 'f87169d89e1943188e1e2acf6be5c063a786f2800ccbbcd2c72b81e077c43709', '/home/n/work/TileIR/public-tileir-blas-20261111/tileir_vllm_post_conv_v2/kernels.py': 'bedcda245f20b1acdc6dcad3df48f8cf6f6d65031325d927a47741fdd2447349', '/home/n/work/TileIR/public-tileir-blas-20261111/gdn_post_conv_qualification_v2/run.py': '1ccbbda45d3131c02a5e20ecab5dc668265811992e03df0f9428cbfaf37619c6', '/home/n/work/TileIR/public-tileir-blas-20261111/regression_v7/modules_norm_conv.py': '8ccee12beaf1ff8104401bc37a98ee7bdce2e0f7f068d887d1a3f395936274e0', '/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/vllm/model_executor/layers/layernorm.py': '7be031e88921fb21661f3b9ee28b4ccb7746405e0ff32ed5895d75b43d3357ed', '/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/vllm/ir/ops/layernorm.py': 'b7621f663a97caf6908507b6cbe03adf23d0001a441d44af2b24c1ad0fc8bf57', '/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/vllm/model_executor/layers/mamba/ops/causal_conv1d.py': 'cb16cc9250c4195c09d6d83b43bba607c2d749b621d8d135a920443d1577265f', '/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/lib/python3.14/site-packages/vllm/third_party/flash_linear_attention/ops/fused_gdn_prefill_post_conv.py': '590717b2107abc90d72548de06b5d315229eb2775c94614f3f545df11cc2b5df'}
def sha(path):
 with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def manifests():
 out={}
 for name,(path,digest)in MANIFESTS.items():
  assert sha(P/path)==digest;out[name]=json.loads((P/path).read_text());assert out[name]['complete']
 return out

def selected(m):
 return [(s,m['gated']['weights']['gated'])for s in m['gated']['samples']]+[(s,m['norm']['weights'][s['weight_key']])for s in m['norm']['samples']if s['phase']=='decode']+[next(s for s in m['conv']['samples']if s['phase']=='prefill'),m['post']['samples'][0]]
def items(tree):
 if isinstance(tree,dict):
  if 'path'in tree and 'sha256'in tree:yield tree
  else:
   for v in tree.values():yield from items(v)
 elif isinstance(tree,(tuple,list)):
  for v in tree:yield from items(v)
def dependencies():
 m=manifests();d={str(P/path):h for path,h in MANIFESTS.values()};d.update(SOURCE_PINS)
 for t in selected(m):
  for x in items(t):
   d[x['path']]=x['sha256']
   if 'storage_path'in x:d[x['storage_path']]=x['storage_sha256']
 return d

def cpu_tests():return ['tileir_vllm_gated_norm_ordered_v1.test_cpu','tileir_indexed_prefill_conv_v1.test_cpu','tileir_vllm_post_conv_v2.test_cpu','regression_v7.test_recurrent_remaining']
def require_metric(c):
 assert c['raw_equal']is True and all(type(c[k])is int and c[k]==0 for k in ('raw_different_elements','strict_failures','reference_nonfinite','candidate_nonfinite'))
 assert c['strict_atol']==c['strict_rtol']==1e-5

def comparison_descriptor(value):
 return dict(shape=list(value.shape),dtype=str(value.dtype),elements=value.numel(),logical_bytes=value.numel()*value.element_size())
def captured_comparison_descriptor(item):
 md=item['metadata'];size={'torch.bfloat16':2,'torch.float32':4}[md['dtype']];elements=math.prod(md['shape'])
 return dict(shape=md['shape'],dtype=md['dtype'],elements=elements,logical_bytes=elements*size)
def actual_comparison(reference,candidate,torch,compare):
 ref=comparison_descriptor(reference);actual=comparison_descriptor(candidate)
 assert ref==actual,'Reference/candidate comparison descriptor mismatch'
 result=compare(reference,candidate,torch)
 result.update(reference_descriptor=ref,actual_descriptor=actual,elements=actual['elements'],logical_bytes=actual['logical_bytes'])
 return result

def negative(value,torch,compare):
 assert value.device.type=='cpu' and value.dtype in (torch.bfloat16,torch.float32) and value.numel()>0
 ref=value.detach().contiguous().reshape(-1)[:32].clone();assert bool(torch.isfinite(ref).all())
 require_metric(compare(ref,ref,torch))
 for kind in ('bit','nan'):
  bad=ref.clone();words=bad.view(torch.int16 if bad.dtype==torch.bfloat16 else torch.int32)
  if kind=='bit':words[0]^=1
  else:words[0]=0x7fc1 if bad.dtype==torch.bfloat16 else 0x7fc00001
  try:require_metric(compare(ref,bad,torch))
  except AssertionError:pass
  else:raise AssertionError('Comparator accepted deliberate '+kind+' corruption')
 return True

def audit(r):
 try:
  assert r['complete']is True and r['context_restored']is True and r['timing']is False and not r.get('error') and not r.get('cleanup_error')
  assert [(x['case'],x['backend'])for x in r['records']]==[(n,b)for n in CASES for b in ('native','v7')]
  specs=selected(manifests())
  for row in r['records']:
   index=CASES.index(row['case']);bundle=specs[index];sample=bundle[0]if index<4 else bundle
   expected=sample['outputs']if index<4 else [sample['output']]if index==4 else [sample['outputs'][k]for k in ('q','k','v','g','beta')]
   assert len(row['descriptors'])==len(expected)
   for got,item in zip(row['descriptors'],expected):
    assert all(got[k]==item['metadata'][k]for k in ('shape','stride','storage_offset','dtype'))
    if 'storage_nbytes'in item['metadata']:assert got['storage_nbytes']==item['metadata']['storage_nbytes']
   assert all(row[k]is True for k in ('readonly','descriptor_exact','fresh_outputs','same_stream','negative_detected','state_metadata_contract'))
   assert len(row['comparisons'])==COUNTS[CASES.index(row['case'])]
   comparison_items=list(expected)+([sample['selected_state_after']]if index==4 else [])
   assert len(comparison_items)==len(row['comparisons'])
   for c,item in zip(row['comparisons'],comparison_items):
    require_metric(c);md=captured_comparison_descriptor(item)
    assert c['reference_descriptor']==c['actual_descriptor']==md
    assert type(c['elements'])is int and c['elements']==md['elements']
    assert type(c['logical_bytes'])is int and c['logical_bytes']==md['logical_bytes']
  return dict(passed=True,cases=6,records=12,full_family_coverage=False)
 except (AssertionError,KeyError,TypeError,ValueError):return dict(passed=False)

def run_gpu(torch,output):
 for path,h in dependencies().items():assert sha(path)==h,('Dependency changed',path)
 from vllm import ir
 from vllm.config import VllmConfig,set_current_vllm_config
 from vllm.model_executor.layers.layernorm import RMSNormGated
 from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
 from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import fused_post_conv_prep as native_post
 from tileir_vllm_gated_norm_ordered_v1 import gated_rms_norm
 from tileir_vllm_gemma_norm_v3 import gemma_rms_norm
 from tileir_indexed_prefill_conv_v1 import causal_conv1d_prefill_indexed
 from tileir_vllm_post_conv_v2 import fused_post_conv_prep
 from gdn_post_conv_qualification_v2.run import compare
 from norm_runtime_qualification_v2.run import identity
 m=manifests();cases=selected(m);assert [s[0]['phase']for s in cases[:2]]==['prefill','decode']
 assert [s[0]['kind']for s in cases[2:4]]==['no_residual','with_residual']
 config=VllmConfig();device=torch.cuda.current_device();prior=torch.cuda.current_stream(device);stream=torch.cuda.Stream(device=device)
 output=prepare_output(output);r=dict(complete=False,timing=False,records=[],scope='six saved boundaries; no full-family claim; prefill state minspan sentinel; default standalone gated config')
 held=[]
 def raw(t):return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
 def digest(t):return hashlib.sha256(raw(t)).hexdigest()
 def load(item):
  t=torch.load(item['path'],map_location='cpu',weights_only=True)
  if 'logical_sha256'in item:assert digest(t)==item['logical_sha256']
  return t
 def desc(t):return dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),dtype=str(t.dtype),storage_nbytes=t.untyped_storage().nbytes())
 def restore(item,memo=None):
  md=item['metadata'];value=load(item)
  if md['device']=='cpu':return value.clone(),None,None
  if 'storage_path'in item:
   key=(md['device'],md['storage_ptr']);host=torch.load(item['storage_path'],map_location='cpu',weights_only=True)
   assert hashlib.sha256(host.numpy().tobytes()).hexdigest()==item['storage_byte_sha256']
   if memo is not None and key in memo:back=memo[key];assert digest(back)==digest(host)
   else:
    back=host.to(device=device)
    if memo is not None:memo[key]=back
   view=torch.as_strided(back.view(value.dtype),md['shape'],md['stride'],md['storage_offset'])
  else:
   host=torch.full((span(md),),False if value.dtype==torch.bool else -7,dtype=value.dtype)
   torch.as_strided(host,md['shape'],md['stride'],md['storage_offset']).copy_(value)
   back=host.to(device=device);view=torch.as_strided(back,md['shape'],md['stride'],md['storage_offset'])
  assert digest(view)==digest(value);return view,back,digest(host)
 def tree(node,memo):
  typ=node['type']
  if typ=='tensor':
   item=node['value'];md=item['metadata'];key=(md.get('device'),md.get('object_identity',item['path']))
   if key not in memo:memo[key]=restore(item)[0]
   return memo[key]
  if typ=='dict':return {tree(k,memo):tree(v,memo)for k,v in node['items']}
  if typ in ('tuple','list'):
   value=[tree(x,memo)for x in node['items']];return tuple(value)if typ=='tuple'else value
  return node['value']
 def check_tree(value,node):
  if node['type']=='tensor':return digest(value)==node['value']['logical_sha256']
  if node['type']=='dict':return len(value)==len(node['items'])and all(check_tree(value[tree(k,{})],v)for k,v in node['items'])
  if node['type']in ('list','tuple'):return type(value).__name__==node['type']and len(value)==len(node['items'])and all(check_tree(v,n)for v,n in zip(value,node['items']))
  return type(value).__name__==node['type']and value==node['value']
 try:
  with torch.cuda.device(device),torch.cuda.stream(stream),torch.inference_mode():
   for index,name in enumerate(CASES):
    bundle=cases[index];s,weight=bundle if index<4 else (bundle,None)
    for backend in ('native','v7'):
     row=dict(case=name,backend=backend,descriptors=[],comparisons=[],readonly=False,descriptor_exact=False,fresh_outputs=False,same_stream=False,negative_detected=False,state_metadata_contract=False);r['records'].append(row)
     memo={};source=dict(s['pre']);
     if weight is not None:source['weight']=weight
     restored={k:restore(v,memo)for k,v in source.items()};inp={k:v[0]for k,v in restored.items()};held[:]=[v[1]for v in restored.values()if v[1]is not None]
     state=None;meta=None
     if index<2:
      gm=m['gated']['modules']['gated'];assert gm['group_size']is None and gm['norm_before_gate']is True and gm['bias']is None and gm['activation']in ('silu','swish')
      if backend=='native':
       with set_current_vllm_config(config):
        mod=RMSNormGated(128,eps=s['epsilon'],group_size=None,norm_before_gate=True,activation=gm['activation'],dtype=inp['weight'].dtype)
        mod.weight=torch.nn.Parameter(inp['weight'],requires_grad=False)
        assert identity(mod.forward_cuda)==s['selected_forward'];out=mod.forward_cuda(inp['x'],inp['z'])
      else:out=gated_rms_norm(inp['x'],inp['z'],inp['weight'],s['epsilon'])
     elif index<4:
      residual=inp.get('residual');event,=s['provider_events'];assert event['provider']=='native'
      op=ir.ops.rms_norm if residual is None else ir.ops.fused_add_rms_norm;fn=op.impls['native'].impl_fn;assert identity(fn)==event['implementation']
      if backend=='native':out=fn(inp['x'],inp['weight'].float()+1.,s['epsilon'])if residual is None else fn(inp['x'],residual,inp['weight'].float()+1.,s['epsilon'])
      else:out=gemma_rms_norm(inp['x'],inp['weight'],s['epsilon'],residual)
     elif index==4:
      sm=s['state_descriptor'];assert span(sm)*2<=700*1024*1024
      state_host=torch.full((span(sm),),-7.,dtype=torch.bfloat16);sv=torch.as_strided(state_host,sm['shape'],sm['stride'],sm['storage_offset']);sv[s['slots']]=load(s['selected_state_before'])
      sb=state_host.to(device=device);held.append(sb);state=torch.as_strided(sb,sm['shape'],sm['stride'],sm['storage_offset'])
      tree_memo={};meta=SimpleNamespace(**{k:tree(v,tree_memo)for k,v in s['metadata_pre'].items()})
      if backend=='native':
       assert identity(causal_conv1d_fn)['source_sha256']==s['helper']['source_sha256']
       out=causal_conv1d_fn(**inp,bias=None,conv_states=state,activation='silu',pad_slot_id=-1,null_block_id=0,block_size_to_align=0,metadata=meta,validate_data=False)
      else:out=causal_conv1d_prefill_indexed(inp['x'],inp['weight'],state,inp['query_start_loc'],inp['cache_indices'],inp['has_initial_state'])
     else:
      assert identity(native_post)['source_sha256']==s['helper']['source_sha256']
      out=(native_post if backend=='native'else fused_post_conv_prep)(**inp,**s['arguments'])
     expected_items=s['outputs']if index<4 else [s['output']]if index==4 else [s['outputs'][k]for k in ('q','k','v','g','beta')]
     expected_tuple=index in (3,5);assert isinstance(out,tuple)==expected_tuple
     outs=list(out)if expected_tuple else[out];assert len(outs)==len(expected_items);held.extend(outs);stream.synchronize()
     for value,item in zip(outs,expected_items):
      md=item['metadata'];actual=desc(value);row['descriptors'].append(actual);assert all(actual[k]==md[k]for k in ('shape','stride','storage_offset','dtype'))
      if 'storage_nbytes'in md:assert actual['storage_nbytes']==md['storage_nbytes']
      c=actual_comparison(load(item),value.detach().cpu(),torch,compare);require_metric(c);row['comparisons'].append(c)
     intervals=[(t.untyped_storage().data_ptr(),t.untyped_storage().data_ptr()+t.untyped_storage().nbytes())for t in [*inp.values(),*outs]+([]if state is None else[state])]
     for j in range(len(inp),len(inp)+len(outs)):
      assert all(j==k or max(intervals[j][0],v[0])>=min(intervals[j][1],v[1])for k,v in enumerate(intervals))
     row['readonly']=all(v[1]is None or digest(v[1])==v[2]for v in restored.values());assert row['readonly']
     if index==4:
      after=load(s['selected_state_after']);sv[s['slots']]=after;assert digest(sb)==digest(state_host)
      c=actual_comparison(after,state[s['slots']].detach().cpu(),torch,compare);require_metric(c);row['comparisons'].append(c)
      expected_meta=s['metadata_post'if backend=='native'else'metadata_pre'];assert all(check_tree(getattr(meta,k),v)for k,v in expected_meta.items())
     row.update(descriptor_exact=True,fresh_outputs=True,same_stream=torch.cuda.current_stream(device)==stream,state_metadata_contract=True)
     row['negative_detected']=all(negative(v.detach().cpu(),torch,compare)for v in outs)
     if state is not None:row['negative_detected'] &= negative(state[s['slots']].detach().cpu(),torch,compare)
     assert row['same_stream'];stream.synchronize();held.clear()
 except BaseException as e:r['error']=repr(e);raise
 finally:
  try:
   stream.synchronize();held.clear();assert torch.cuda.current_device()==device and torch.cuda.current_stream(device)==prior;r['context_restored']=True
   if 'error'not in r:r['complete']=True;assert audit(r)['passed']
  except BaseException as e:r['complete']=False;r['cleanup_error']=repr(e);raise
  finally:save_partial(output,r)
 return r
