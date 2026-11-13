"""DRAFT: one captured prefill bank plus private synthetic parent state.
No decode repeat, no full live shared-owner or 24-layer qualification claim.
"""
import hashlib,json
from pathlib import Path
P=Path('/home/n/work/TileIR/public-tileir-blas-20261111')
MANIFEST=P/'gdn_chunk_runtime_capture_v1/gdnChunkCapture01/bank/MANIFEST.json'
STAGES=P/'gdn_prefill_stage_capture_v1/gdnStagesCapture01/RESULT.json'
PARENT=P/'gdn_parent_state_synthetic_v2/MANIFEST.json'
FIXED={str(MANIFEST):'2523a13367f85d7590c13d43045c492a7d224bca4d6c5d8f1a9996fa0c5248b1',str(STAGES):'d3f56ccfece2d0a2ae8d4c221fc40aedaebe66dbc1cf967016ae4f7b0568d699',str(PARENT):'3bf53c99321c631d5ae4eed492c2391fb5c3ea7ca41f443e0adb497f54358cb4'}
LABELS=('g_cumsum','A_kkt','A_inverse','w','u','h','v_new','final_state','o')
SOLVER='tileir_gdn_solve_tril_v6' # Original v7, NOT old standalone v5 qualifier.
def cpu_tests():return ['regression_v7.test_gdn_prefill']
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dependencies():
 for p,h in FIXED.items():assert sha(p)==h
 m=json.loads(MANIFEST.read_text());s,=m['samples'];cap=json.loads(STAGES.read_text())
 assert m['complete'] and m['observers_removed'] and cap['complete'] and cap['artifacts_written']
 assert all(s['input_logical_preserved'].values())
 assert s['arguments']==dict(output_final_state=True,use_qk_l2norm_in_kernel=False,core_attn_out=None,aiter_prefill_metadata=None)
 result=dict(FIXED)
 for e in [*s['pre'].values(),*s['outputs'].values(),*cap['artifacts'].values()]:result[e['path']]=e['sha256']
 return result

def describe(t):
 return dict(shape=list(t.shape),stride=list(t.stride()),dtype=str(t.dtype),offset=t.storage_offset(),numel=t.numel(),owner_bytes=t.untyped_storage().nbytes())
def expected_descriptors():
 import math
 result={}
 for label,e in json.loads(STAGES.read_text())['artifacts'].items():
  numel=math.prod(e['shape']);size={'torch.float32':4,'torch.bfloat16':2}[e['dtype']]
  result[label]=dict(shape=e['shape'],stride=e['stride'],dtype=e['dtype'],offset=0,numel=numel,owner_bytes=numel*size)
 return result

def compare_snapshot(torch,reference,candidate):
 assert reference.device.type==candidate.device.type=='cpu'
 a=reference.contiguous().view(torch.uint8);b=candidate.contiguous().view(torch.uint8)
 assert a.shape==b.shape
 raw_mismatches=int((a!=b).sum().item())
 x=reference.float();y=candidate.float();strict=int(((x-y).abs()>1e-5+1e-5*x.abs()).sum().item())
 return dict(reference_descriptor=describe(reference),candidate_descriptor=describe(candidate),bytes_compared=a.numel(),raw_byte_mismatches=raw_mismatches,raw_equal=raw_mismatches==0,strict_failures=strict,reference_nonfinite=int((~torch.isfinite(reference)).sum().item()),candidate_nonfinite=int((~torch.isfinite(candidate)).sum().item()))
def require_comparison(m,descriptor):
 assert m['reference_descriptor']==m['candidate_descriptor']==descriptor
 for k in ('bytes_compared','raw_byte_mismatches','strict_failures','reference_nonfinite','candidate_nonfinite'):assert type(m[k])is int
 assert m['bytes_compared']==descriptor['owner_bytes'] and m['raw_equal']is True
 assert m['raw_byte_mismatches']==m['strict_failures']==m['reference_nonfinite']==m['candidate_nonfinite']==0
 return m
def snapshot_check(torch,reference,candidate,descriptor):
 return require_comparison(compare_snapshot(torch,reference,candidate),descriptor)
def negative_probe(torch,reference,label,checker=snapshot_check):
 descriptor=describe(reference);positive=checker(torch,reference,reference.clone(),descriptor);records=[]
 for case in ('bitflip','finite_large'):
  bad=reference.clone()
  if case=='bitflip':bad.view(torch.uint8).flatten()[0]^=1
  else:bad.flatten()[0]=10000 if float(reference.flatten()[0])!=10000 else -10000
  metrics=compare_snapshot(torch,reference,bad)
  try:checker(torch,reference,bad,descriptor)
  except AssertionError:rejected=True
  else:raise RuntimeError('Corrupted '+label+' accepted by actual snapshot checker')
  assert metrics['raw_byte_mismatches']>0
  if case=='finite_large':assert metrics['strict_failures']>0 and metrics['candidate_nonfinite']==0
  records.append(dict(label=label,case=case,rejected=rejected,comparison=metrics))
 return dict(label=label,positive=positive,negative=records)

def expected_order():
 fixtures=json.loads(PARENT.read_text())['fixtures']
 return [('core_layer0',a,i) for a in ('native','tile') for i in (0,1)]+[(f['name'],a,i) for f in fixtures for a in ('native','tile') for i in (0,1)]

def audit(r):
 try:
  assert r['complete'] is True and r['solver']==SOLVER and r['timing'] is False
  assert [(x['case'],x['arm'],x['repeat'])for x in r['records']]==expected_order()
  assert r['scope']==dict(prefill_actual_banks=1,parent_synthetic_fixtures=3,live_shared_owner=False,all_24_layers=False)
  descriptors=expected_descriptors()
  for row in r['records']:
   assert row['passed'] is True and row['readonly'] is True and row['repeat_raw'] is True and row['stream_unchanged'] is True
   if row['case']=='core_layer0':
    assert row['stages']==list(LABELS) and row['raw_all_stages'] and row['raw_output_state'] and row['finite'] and row['fresh_nonalias'] and row['observer_restored']
    assert [x['label']for x in row['stage_metrics']]==list(LABELS)
    for item in row['stage_metrics']:require_comparison(item['comparison'],descriptors[item['label']])
    assert set(row['output_metrics'])=={'o','final_state'}
    for label,metric in row['output_metrics'].items():require_comparison(metric,descriptors[label])
   else:
    for k in ('full_private_backing_raw','gather_raw','invert_raw','mask_raw','scatter_return_identity','mask_return_identity','fresh_nonalias','false_positive_zero','special_bits'):
     assert row[k] is True
  assert r['raw_negative_detected'] is True
  assert [x['label']for x in r['negative_probes']]==['o','final_state']
  for probe in r['negative_probes']:
   descriptor=descriptors[probe['label']];require_comparison(probe['positive'],descriptor)
   assert [(x['label'],x['case'])for x in probe['negative']]==[(probe['label'],k)for k in ('bitflip','finite_large')]
   for negative in probe['negative']:
    metric=negative['comparison'];assert negative['rejected']is True
    assert metric['reference_descriptor']==metric['candidate_descriptor']==descriptor
    for key in ('bytes_compared','raw_byte_mismatches','strict_failures','reference_nonfinite','candidate_nonfinite'):assert type(metric[key])is int
    assert metric['bytes_compared']==descriptor['owner_bytes']
    assert type(metric['raw_byte_mismatches'])is int and metric['raw_byte_mismatches']>0
    assert metric['raw_equal']is False and metric['reference_nonfinite']==metric['candidate_nonfinite']==0
    if negative['case']=='finite_large':assert type(metric['strict_failures'])is int and metric['strict_failures']>0
  return dict(passed=True)
 except (KeyError,AssertionError,TypeError):return dict(passed=False)

def run_gpu(torch,output):
 for p,h in dependencies().items():assert sha(p)==h
 from gdn_prefill_chain_qualification_v1.chain import execute
 from gdn_prefill_stage_capture_v1.observer import Observer
 from norm_runtime_contract_capture_v1.observer import identity
 from tileir_gdn_chunk_cumsum_v1 import chunk_local_cumsum
 from tileir_gdn_chunk_kkt_v5 import chunk_scaled_dot_kkt
 from tileir_gdn_solve_tril_v6 import solve_tril
 from tileir_gdn_wy_v1 import recompute_w_u_fwd
 from tileir_gdn_chunk_state_v1 import chunk_state
 from tileir_gdn_chunk_output_v2 import chunk_output
 import importlib
 chunk=importlib.import_module('vllm.third_party.flash_linear_attention.ops.chunk')
 solver=importlib.import_module('vllm.third_party.flash_linear_attention.ops.solve_tril')
 assert solver.FLA_TRIL_PRECISION=='ieee' and solver.is_tma_supported is False
 ops=dict(cumsum=chunk_local_cumsum,kkt=chunk_scaled_dot_kkt,solve=solve_tril,wy=recompute_w_u_fwd,state=chunk_state,output=chunk_output)
 def raw(t):return t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
 def digest(t):return hashlib.sha256(raw(t)).hexdigest()
 def disjoint(a,b):
  x,y=a.untyped_storage(),b.untyped_storage();return x.data_ptr()+x.nbytes()<=y.data_ptr() or y.data_ptr()+y.nbytes()<=x.data_ptr()
 def load(e):
  t=torch.load(e['path'],map_location='cpu',weights_only=True);assert digest(t)==e['logical_sha256'];return t
 s,=json.loads(MANIFEST.read_text())['samples'];cap=json.loads(STAGES.read_text())
 cpus={k:load(e)for k,e in s['pre'].items()};expected={k:load(e)for k,e in cap['artifacts'].items()};full={k:load(e)for k,e in s['outputs'].items()}
 assert tuple(expected)==LABELS or set(expected)==set(LABELS)
 for k,t in cpus.items():
  d=s['pre'][k]['metadata'];assert list(t.shape)==d['shape'] and list(t.stride())==d['stride'] and t.storage_offset()==0 and t.untyped_storage().nbytes()==t.numel()*t.element_size()==d['storage_nbytes']<=128*1024**2
 r=dict(complete=False,solver=SOLVER,timing=False,scope=dict(prefill_actual_banks=1,parent_synthetic_fixtures=3,live_shared_owner=False,all_24_layers=False),records=[],raw_negative_detected=False)
 prior={};negative_refs={};current=torch.cuda.current_stream();descriptors=expected_descriptors()
 with torch.inference_mode():
  for arm in ('native','tile'):
   for repeat in (0,1):
    inputs={k:t.cuda() for k,t in cpus.items()};retained={};seen=[];stage_metrics=[];obs=None
    def snapshot(t,label):
     e=expected[label];assert t.shape==e.shape and t.dtype==e.dtype and t.stride()==e.stride() and t.storage_offset()==0 and t.untyped_storage().nbytes()==t.numel()*t.element_size()<=128*1024**2
     assert all(disjoint(t,v) for v in [*inputs.values(),*retained.values()])
     assert describe(t)==descriptors[label]
     metric=snapshot_check(torch,e,t.cpu(),descriptors[label]);stage_metrics.append(dict(label=label,comparison=metric))
     assert all(raw(v)==raw(cpus[k]) for k,v in inputs.items())
     assert all(raw(v)==raw(expected[k]) for k,v in retained.items())
     seen.append(label);retained[label]=t;return dict(passed=True)
    try:
     if arm=='native':
      obs=Observer(chunk,snapshot,identity);obs.install()
      try:out=chunk.chunk_gated_delta_rule(**inputs,output_final_state=True,use_qk_l2norm_in_kernel=False,core_attn_out=None)
      finally:obs.close()
      assert obs.complete()
     else:out=execute(inputs,ops,lambda outputs:[snapshot(v,k)for k,v in outputs.items()])
     assert seen==list(LABELS) and isinstance(out,tuple) and len(out)==2
     output_metrics={k:snapshot_check(torch,full[k],t.cpu(),descriptors[k])for k,t in zip(('o','final_state'),out)}
     if arm=='native' and repeat==0:negative_refs={k:t.detach().cpu()for k,t in zip(('o','final_state'),out)}
     blobs=tuple(raw(t)for t in out)
     assert repeat==0 or prior[arm]==blobs;prior[arm]=blobs
     assert all(raw(v)==raw(cpus[k]) for k,v in inputs.items())
     assert torch.cuda.current_stream()==current
     r['records'].append(dict(case='core_layer0',arm=arm,repeat=repeat,passed=True,readonly=True,repeat_raw=True,stream_unchanged=True,stages=seen,stage_metrics=stage_metrics,output_metrics=output_metrics,raw_all_stages=True,raw_output_state=True,finite=True,fresh_nonalias=True,observer_restored=obs is None or obs.closed))
    finally:
     if obs is not None and obs.bindings:obs.close()
    del inputs,retained,out
  assert prior['native']==prior['tile']
  r['negative_probes']=[negative_probe(torch,negative_refs[label],label)for label in ('o','final_state')];r['raw_negative_detected']=True
  del prior,negative_refs,expected,full,cpus
  _parent(torch,r,raw,disjoint)
 torch.cuda.synchronize();r['complete']=True;assert audit(r)['passed'];return r

def _parent(torch,r,raw,disjoint):
 from tileir_gdn_parent_state_v2 import gather_state,mask_initial_state_,scatter_state_
 from tileir_vllm_bool_not_v1.api import bitwise_not_bool32
 fixtures=json.loads(PARENT.read_text())['fixtures'];current=torch.cuda.current_stream()
 def inject(t):
  p=torch.tensor([0,0x80000000,0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x7f812345],dtype=torch.uint32,device='cuda');t.view(torch.uint32).flatten()[:7]=p
 for f in fixtures:
  reference=None
  for arm in ('native','tile'):
   for repeat in (0,1):
    n,h,k,v=f['shape'];inner=h*k*v;length=f['offset']+(n-1)*f['pitch']+inner+97
    assert length*4<=384*1024**2
    backing=torch.empty(length,device='cuda',dtype=torch.float32)
    for lo in range(0,length,1024*1024):
     ix=torch.arange(lo,min(length,lo+1024*1024),device='cuda',dtype=torch.int64);values=((ix%1021)+0x3f000000).to(torch.int32)
     values=torch.where(ix%7==0,-2147483648,values);values=torch.where(ix%11==0,0,values);backing.view(torch.int32)[lo:lo+len(ix)]=values
    owner=backing.as_strided(tuple(f['shape']),(f['pitch'],k*v,v,1),f['offset']);inject(backing)
    for row in f['indices'][::f['index_stride']]:inject(owner[row])
    ib=torch.tensor(f['indices'],device='cuda',dtype=torch.int64 if f['index64']else torch.int32);indices=ib[::f['index_stride']]
    mask=torch.tensor(f['mask'],device='cuda',dtype=torch.bool);before=raw(backing);ibefore=raw(ib);mbefore=raw(mask)
    gathered=owner[indices] if arm=='native'else gather_state(owner,indices)
    assert disjoint(gathered,owner) and gathered.is_contiguous() and gathered.storage_offset()==0
    graw=raw(gathered);assert raw(backing)==before
    inv=~mask if arm=='native'else bitwise_not_bool32(mask)
    assert raw(inv)==raw(~mask) and disjoint(inv,mask)
    if arm=='native':gathered[inv,...]=0
    else:assert mask_initial_state_(gathered,mask)is gathered
    assert raw(backing)==before
    for row,keep in enumerate(f['mask']):
     if not keep:assert bool((gathered[row].view(torch.int32)==0).all())
    last=torch.full_like(gathered,1.25);last.view(torch.int32).flatten()[::5]=-2147483648
    for row in range(last.shape[0]):inject(last[row])
    lastbefore=raw(last)
    if arm=='native':owner[indices]=last
    else:assert scatter_state_(owner,indices,last)is owner
    assert raw(last)==lastbefore and raw(ib)==ibefore and raw(mask)==mbefore and torch.cuda.current_stream()==current
    values=(raw(backing),graw,raw(gathered),raw(inv))
    if reference is None:reference=values
    else:assert values==reference
    r['records'].append(dict(case=f['name'],arm=arm,repeat=repeat,passed=True,readonly=True,repeat_raw=True,stream_unchanged=True,full_private_backing_raw=True,gather_raw=True,invert_raw=True,mask_raw=True,scatter_return_identity=True,mask_return_identity=True,fresh_nonalias=True,false_positive_zero=True,special_bits=True))
    del backing,owner,ib,indices,mask,gathered,inv,last,values,before,lastbefore
