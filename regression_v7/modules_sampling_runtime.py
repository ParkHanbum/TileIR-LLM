"""External draft: original-v7 leaf API samples, not full sampler/runtime coverage.
GPU requires the regression coordinator's explicit bounded entry. No import GPU work.
"""
import hashlib,json
from pathlib import Path
P=Path('/home/n/work/TileIR/public-tileir-blas-20261111')
MANIFESTS={
 'greedy_sampler_runtime_capture_v1/greedySamplerCapture01/bank/MANIFEST.json':'8aff9ec850b5df8dcb1c08621e1328cf5d7fbfcddbaa78323616e0217d9d6b72',
 'tp1_embedding_runtime_capture_v2/tp1EmbeddingCapture01/bank/MANIFEST.json':'bf6f73267ada00e9d8ed8f32b1b578620388e26a61d249443a724427efb34048'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def fixtures():
 values=[]
 for rel,h in MANIFESTS.items():
  assert sha(P/rel)==h,'Historical fixture manifest drift'
  values.append(json.loads((P/rel).read_text()))
 from greedy_sampler_qualification_v1.run import check_manifest
 from tp1_embedding_qualification_v1.run import check
 greedy=check_manifest(values[0]);assert check(values[1])
 emb=sorted(values[1]['samples'],key=lambda s:0 if s['phase']=='prefill'else 1)
 assert [s['phase']for s in emb]==['prefill','decode']
 return greedy,emb
def sampler_cases():
 rel=next(iter(MANIFESTS));assert sha(P/rel)==MANIFESTS[rel]
 m=json.loads((P/rel).read_text())
 from selected_ranks_qualification_v1.run import check_manifest
 return check_manifest(m)
def dependencies():
 g,e=fixtures();out={str(P/k):v for k,v in MANIFESTS.items()}
 capture=json.loads((P/next(iter(MANIFESTS))).read_text())
 for entry in capture['source'].values():out[entry['source']]=entry['source_sha256']
 for s in g:
  for x in [*s['inputs'].values(),s['outputs']['sampled']]:out[x['path']]=x['sha256']
 for sample in sampler_cases():
  for x in [*sample['inputs'].values(),*sample['outputs'].values()]:out[x['path']]=x['sha256']
 for s in e:
  for k in ('input','unique_ids','inverse','selected_weight_rows','output'):out[s[k]['path']]=s[k]['sha256']
 return out
def cpu_tests():
 return ['tileir_vllm_greedy_sample_v1.test_cpu','tileir_vllm_embedding_int64_v1.test_cpu','runtime_metadata_five_v1.test_cpu','lean_public_composition_v1.test_sampler_cpu','tileir_vllm_top5_v2.test_cpu','selected_logprobs_ordered_qualification_v1.test_cpu','tileir_vllm_selected_ranks_v1.test_cpu','regression_v7.test_sampling_runtime']
def expected_rows():
 return [('greedy',str(i),'saved_actual')for i in (0,15)]+[(f,str(i),'saved_actual')for i in (0,15)for f in ('top5','logprobs','ranks')]+[('embedding',s,'derived_selected_row_table')for s in ('prefill','decode')]+[('widen',str(n),'synthetic')for n in (32,1696)]+[('subtract',str(o),'synthetic')for o in (0,1)]+[('positive','32','synthetic'),('full','2048','synthetic')]
def expected_descriptor(family,case):
 if family in ('greedy','ranks'):shape,dtype=[32],'torch.int64'
 elif family=='top5':shape,dtype=[32,5],'torch.int64'
 elif family=='logprobs':shape,dtype=[32,6],'torch.float32'
 elif family=='embedding':shape,dtype=[1696 if case=='prefill' else 32,2560],'torch.bfloat16'
 elif family=='widen':shape,dtype=[int(case)],'torch.int64'
 elif family=='subtract':shape,dtype=[32],'torch.int32'
 elif family=='positive':shape,dtype=[32],'torch.bool'
 elif family=='full':shape,dtype=[2048],'torch.int32'
 else:raise ValueError('Unknown family')
 n=1
 for x in shape:n*=x
 return dict(shape=shape,stride=[shape[1],1] if len(shape)==2 else [1],dtype=dtype,numel=n,storage_offset=0)
def descriptor(x):
 return dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),numel=x.numel(),storage_offset=x.storage_offset())
IO_BUDGET=dict(embedding_table_bytes=1271398400,embedding_full_table_D2H_bytes=7628390400,embedding_owner_scans_per_case=3,embedding_cases=2,raw_materialization='one full table per scan; CPU byte buffer ~1.27GB plus transient copies; not a low-memory test',fixture_scope='Derived zero table plus saved selected rows, not full checkpoint',disk_writes='RESULT.json only; no tensor output dump',budget_is='known table traffic lower bound; other owners/outputs and fixture file reads additional')
FIELDS=('raw_equal','readonly','layout','fresh_disjoint','current_stream','finite','passed')
def audit(r):
 try:
  rows=r['records'];assert r['complete']is True and len(rows)==16
  for row,(family,case,origin)in zip(rows,expected_rows()):
   assert (row['family'],row['case'],row['origin'])==(family,case,origin)
   assert all(row[k]is True for k in FIELDS)
   want=expected_descriptor(family,case)
   assert row['output_descriptor']==want and row['expected_descriptor']==want
  return True
 except (KeyError,TypeError,AssertionError):return False
def run_gpu(torch,output):
 for p,h in dependencies().items():assert sha(p)==h,'Fixture bytes changed'
 from tileir_vllm_greedy_sample_v1 import greedy_sample
 from tileir_vllm_tp1_embedding_v1 import embedding
 from tileir_vllm_embedding_int64_v1.api import embedding_ids_long
 from runtime_metadata_five_v1 import api as meta
 from tileir_vllm_top5_v2 import top5_indices
 from tileir_vllm_selected_logprobs_v2 import compute_token_logprobs
 from tileir_vllm_selected_ranks_v1 import selected_token_ranks
 from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
 from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs as native_logprobs, _ranks_kernel
 g,e=fixtures();output=Path(output);output.mkdir(parents=True,exist_ok=True)
 result=dict(complete=False,records=[],io_budget=IO_BUDGET,scope='2 saved greedy logits; 2 saved BF16 top5/logprob/rank banks; 2 selected-row embedding tables; 6 synthetic metadata/cast cases. No full model, RNG, general tie policy, call6/call8 or full checkpoint claim.',lifetime='Synchronous isolated calls; no asynchronous owner-retirement or concurrency proof')
 def save():(output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 def raw(x):return x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
 def load(item):
  x=torch.load(item['path'],weights_only=True,map_location='cpu');assert hashlib.sha256(raw(x)).hexdigest()==item['logical_sha256'];return x
 def inspect(family,case,origin,fn,expected,owners,native=None):
  want=expected_descriptor(family,case);assert descriptor(expected)==want,'Expected output descriptor drift'
  before=[hashlib.sha256(raw(x)).hexdigest()for x in owners];stream=torch.cuda.current_stream().cuda_stream
  if native is not None:
   reference=native();torch.cuda.synchronize();assert descriptor(reference)==want,'Native output descriptor drift'
   assert raw(reference)==raw(expected),'Native output drift'
   assert before==[hashlib.sha256(raw(x)).hexdigest()for x in owners],'Native modified input'
  got=fn();torch.cuda.synchronize()
  row=dict(family=family,case=case,origin=origin,output_descriptor=descriptor(got),expected_descriptor=descriptor(expected),raw_equal=raw(got)==raw(expected),readonly=before==[hashlib.sha256(raw(x)).hexdigest()for x in owners],layout=(descriptor(got)==want),fresh_disjoint=all(got.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()for x in owners),current_stream=torch.cuda.current_stream().cuda_stream==stream,finite=bool(torch.isfinite(got).all()),passed=False)
  row['passed']=all(row[k]is True for k in FIELDS if k!='passed');result['records'].append(row);save();assert row['passed'],'First mismatch HOLD'
 try:
  with torch.inference_mode():
   for s in g:
    item=s['inputs']['logits'];m=item['metadata'];assert m['shape']==[32,248320]and m['stride']==[248320,1]and m['storage_offset']==0 and m['storage_nbytes']==31784960
    values={}
    for key,entry in s['inputs'].items():
     cpu=load(entry);d=entry['metadata'];assert list(cpu.shape)==d['shape'] and list(cpu.stride())==d['stride'] and d['storage_offset']==0 and cpu.numel()*cpu.element_size()==d['storage_nbytes']
     values[key]=cpu.cuda()
    x=values['logits'];expected=load(s['outputs']['sampled'])
    inspect('greedy',str(s['call_index']),'saved_actual',lambda:greedy_sample(x),expected,list(values.values()),native=lambda:gumbel_sample(**values,**s['observations']['scalars']));del x,expected,values
   for s in sampler_cases():
    x=load(s['inputs']['logits']).cuda();chosen=load(s['inputs']['sampled_token_ids']).cuda();ids=load(s['outputs']['logprob_token_ids']).cuda()
    assert x.shape==(32,248320) and x.stride()==(248320,1) and x.dtype==torch.bfloat16
    assert ids.shape==(32,6) and ids.stride()==(6,1) and chosen.shape==(32,) and chosen.stride()==(1,)
    expected_ids=ids[:,1:].contiguous().cpu()
    inspect('top5',str(s['call_index']),'saved_actual',lambda:top5_indices(x),expected_ids,[x],native=lambda:torch.topk(x,5,dim=-1,largest=True,sorted=True).indices)
    inspect('logprobs',str(s['call_index']),'saved_actual',lambda:compute_token_logprobs(x,ids),load(s['outputs']['logprobs']),[x,ids],native=lambda:native_logprobs(x,ids))
    def native_ranks():
     out=torch.empty((32,),dtype=torch.int64,device=x.device);_ranks_kernel[(32,)](out,x,x.stride(0),chosen,248320,BLOCK_SIZE=8192);return out
    inspect('ranks',str(s['call_index']),'saved_actual',lambda:selected_token_ranks(x,chosen),load(s['outputs']['selected_token_ranks']),[x,chosen],native=native_ranks)
    del x,chosen,ids
   for s in e:
    ids,unique,inverse,rows,expected=[load(s[k])for k in ('input','unique_ids','inverse','selected_weight_rows','output')]
    assert torch.equal(unique[inverse],ids)and raw(rows[inverse])==raw(expected)
    assert bool(((ids>=0)&(ids<248320)).all())and unique.unique().numel()==unique.numel()
    ids=ids.cuda();table=torch.zeros((248320,2560),dtype=torch.bfloat16,device='cuda');table.index_copy_(0,unique.cuda(),rows.cuda())
    inspect('embedding',s['phase'],'derived_selected_row_table',lambda:embedding(ids,table),expected,[ids,table],native=lambda:torch.nn.functional.embedding(ids,table));del table,ids,expected
   vals=[-2147483648,-1,0,1,2147483647]
   for n in (32,1696):
    x=torch.tensor((vals*((n+4)//5))[:n],dtype=torch.int32,device='cuda');inspect('widen',str(n),'synthetic',lambda:embedding_ids_long(x),x.to(torch.int64),[x])
   for offset in (0,1):
    a=torch.tensor((vals*7)[:33],dtype=torch.int32,device='cuda');b=torch.tensor((list(reversed(vals))*7)[:33],dtype=torch.int32,device='cuda');av=a[offset:offset+32];bv=b[:32]
    inspect('subtract',str(offset),'synthetic',lambda:meta.subtract(av,bv),av-bv,[a,b])
   x=torch.tensor((vals*7)[:32],dtype=torch.int32,device='cuda');inspect('positive','32','synthetic',lambda:meta.positive(x),x>0,[x])
   expected=torch.full((2048,),-1,dtype=torch.int32,device='cuda');inspect('full','2048','synthetic',lambda:meta.full((2048,),-1,dtype=torch.int32,device=expected.device),expected,[])
  result['complete']=True;assert audit(result);save();return result
 except BaseException as exc:
  result['complete']=False;result['error']=type(exc).__name__+': '+str(exc);save();raise
