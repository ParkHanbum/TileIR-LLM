"""Contract-shaped synthetic tests for original-v7 runner helpers, not actual-bank replay.
Explicit coordinator GPU entry only. Transport failure retains all pending owners.
No new numeric implementation; native torch references run outside candidate APIs.
"""
import hashlib,json
from pathlib import Path
P=Path('/home/n/work/TileIR/public-tileir-blas-20261111')
_PENDING_TRANSPORTS=[]
MODULES=('tileir_vllm_runner_constructors_v1','tileir_vllm_runner_bool_fill_v1','runner_scalar32_qualification_v3','tileir_vllm_bool_not_v1','tileir_vllm_mamba_accepted_scatter_v1','batch_ptr_int_cast_v1','tileir_vllm_slot_mapping_v1','tileir_vllm_pos_seq_v1','tileir_staged_typed_v2','zero_kv_phase2_typed_plan_v1','zero_kv_typed_lean_v1')
def cpu_tests():return ['regression_v7.test_runtime_remaining','tileir_vllm_pos_seq_v1.test_cpu','tileir_vllm_slot_mapping_v1.test_cpu','runner_constructors_qualification_v1.test_cpu','tileir_vllm_runner_bool_fill_v1.test_cpu','runner_scalar32_qualification_v3.test_cpu','bool_not_qualification_v1.test_cpu','tileir_vllm_mamba_accepted_scatter_v1.test_cpu','batch_ptr_int_cast_v1.test_cpu','pos_seq_runtime_qualification_v1.test_cpu','tileir_staged_typed_v2.test_contract','zero_kv_phase2_typed_plan_v1.test_cpu']
def dependencies():
 out={}
 for name in MODULES:
  for f in (P/name).glob('*.py'):out[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
 return out
def expected_rows():
 return [('construct_arange','32'),('construct_zero','32')]+[('padding',str(n))for n in (0,32,1696)]+[('scalar_fill',str(i))for i in (0,31)]+[('bool_not','32'),('accepted_scatter','unique'),('accepted_scatter','negative_skip'),('batch_cast','224'),('slot_mapping','two_groups'),('pos_seq','32')]+[('staged',d)for d in ('int32','int64','float32')]+[('zero_words','32')]
def expected_descriptor(family,case):
 if family=='zero_words':return dict(shape=[32],stride=[1],dtype='torch.int32',numel=32,storage_offset=16)
 if family=='slot_mapping':return dict(shape=[2,32],stride=[40,1],dtype='torch.int64',numel=64,storage_offset=0)
 if family=='staged':return dict(shape=[4,8],stride=[8,1],dtype='torch.'+case,numel=32,storage_offset=0)
 if family=='pos_seq':return dict(shape=[40],stride=[1],dtype='torch.int64',numel=40,storage_offset=0)
 n=33 if family=='construct_arange' else (int(case)if family=='padding'else 224 if family=='batch_cast'else 32)
 dtype='torch.bool'if family in ('padding','bool_not')else 'torch.int32'
 offset=3 if family=='padding'else 0
 return dict(shape=[n],stride=[1],dtype=dtype,numel=n,storage_offset=offset)
def descriptor(x):return dict(shape=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),numel=x.numel(),storage_offset=x.storage_offset())
def return_kind(family):
 if family=='slot_mapping':return 'alias_storage'
 return 'same_view'if family in ('padding','batch_cast')else 'none'if family in ('scalar_fill','accepted_scatter','staged','pos_seq','zero_words')else 'fresh'
GATES=('raw_equal','whole_owner_equal','readonly','return_contract','stream','passed')
def audit(r):
 try:
  assert r['complete']is True and len(r['records'])==17 and r['pending_transports']==0
  for row,(family,case)in zip(r['records'],expected_rows()):
   assert (row['family'],row['case'],row['value_source'])==(family,case,'contract_shaped_synthetic')
   assert row['descriptor']==expected_descriptor(family,case)and row['return_kind']==return_kind(family)
   assert all(row[k]is True for k in GATES)
  t=r['transport'];assert t['transfer_attempts']==t['transfer_returns']==t['cast_launch_attempts']==t['cast_launch_returns']==1
  assert t['temporary_bytes']==1792 and t['same_dtype_transfer']is True and t['source_pinned']is True
  return True
 except (KeyError,TypeError,AssertionError):return False
def run_gpu(torch,output):
 from tileir_vllm_runner_constructors_v1.api import cu_num_logits,expanded_local_pos
 from tileir_vllm_runner_bool_fill_v1.api import fill_padding_
 from runner_scalar32_qualification_v3.api import fill_request_slot
 from tileir_vllm_bool_not_v1.api import bitwise_not_bool32
 from tileir_vllm_mamba_accepted_scatter_v1.api import scatter_num_accepted_
 from batch_ptr_int_cast_v1.api import Transport
 from tileir_vllm_slot_mapping_v1.api import compute_slot_mappings,reference_cpu
 from tileir_vllm_pos_seq_v1.api import prepare_pos_seq_lens,reference_checked
 from tileir_staged_typed_v2.api import apply as staged_apply
 assert not _PENDING_TRANSPORTS,'Unreleased earlier transport'
 output=Path(output);output.mkdir(parents=True,exist_ok=True)
 r=dict(complete=False,records=[],pending_transports=0,scope='17 synthetic helper cases at original-v7 observed shapes; no model hooks or UVA/actual zero-KV owner registration coverage; zero leaf uses bounded private typed owner',transport_scope='1792-byte same-dtype H2D is runtime transport, not TileIR compute; 224 low32 casts only; whole destination8192 bytes checked')
 def save():(output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 def raw(x):return x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
 def record(family,case,fn,reference,owners=(),view=None,whole=None,expected_whole=None):
  before=[raw(x)for x in owners];stream=torch.cuda.current_stream().cuda_stream
  returned=fn();torch.cuda.synchronize();got=returned if view is None else view;kind=return_kind(family)
  ret=(returned.untyped_storage().data_ptr()==whole.untyped_storage().data_ptr())if kind=='alias_storage'else (returned is view)if kind=='same_view'else (returned is None)if kind=='none'else all(returned.untyped_storage().data_ptr()!=x.untyped_storage().data_ptr()for x in owners)
  row=dict(family=family,case=case,value_source='contract_shaped_synthetic',descriptor=descriptor(got),return_kind=kind,raw_equal=raw(got)==raw(reference),whole_owner_equal=True if whole is None else raw(whole)==raw(expected_whole),readonly=before==[raw(x)for x in owners],return_contract=ret,stream=stream==torch.cuda.current_stream().cuda_stream,passed=False)
  row['passed']=all(row[k]is True for k in GATES if k!='passed')and row['descriptor']==expected_descriptor(family,case);r['records'].append(row);save();assert row['passed'],'Helper mismatch HOLD'
 try:
  with torch.inference_mode():
   device=torch.device('cuda:0')
   record('construct_arange','32',lambda:cu_num_logits(32,device),torch.arange(33,dtype=torch.int32,device=device))
   record('construct_zero','32',lambda:expanded_local_pos(32,device),torch.zeros(32,dtype=torch.int32,device=device))
   for n in (0,32,1696):
    owner=torch.ones(n+7,dtype=torch.bool,device=device);view=owner[3:3+n];expected=owner.clone();expected[3:3+n].fill_(False)
    record('padding',str(n),lambda:fill_padding_(view,False),expected[3:3+n],view=view,whole=owner,expected_whole=expected)
   for index in (0,31):
    owner=torch.full((32,),-123,dtype=torch.int32,device=device);expected=owner.clone();expected[index].fill_(1)
    record('scalar_fill',str(index),lambda:fill_request_slot(owner,index),expected,view=owner,whole=owner,expected_whole=expected)
   mask=torch.tensor([False,True]*16,dtype=torch.bool,device=device);record('bool_not','32',lambda:bitwise_not_bool32(mask),torch.bitwise_not(mask),owners=(mask,))
   for case in ('unique','negative_skip'):
    ids=torch.arange(32,dtype=torch.int64,device=device)
    if case=='negative_skip':ids[::4]=-1
    counts=torch.arange(32,dtype=torch.int32,device=device);accepted=torch.full((32,),-9,dtype=torch.int32,device=device);expected=accepted.clone();valid=ids>=0;expected[ids[valid]]=torch.clamp(counts[valid],min=1)
    record('accepted_scatter',case,lambda:scatter_num_accepted_(ids,counts,accepted),expected,owners=(ids,counts),view=accepted,whole=accepted,expected_whole=expected)
   src=torch.empty(224,dtype=torch.int64,pin_memory=True);values=[-2**63,2**63-1,-2**32,2**32,-2**31,2**31-1,0,-1];src.copy_(torch.tensor(values*28,dtype=torch.int64));owner=torch.full((2048,),-12345,dtype=torch.int32,device=device);dst=owner[:224];expected=owner.clone();expected[:224].copy_(src)
   transport=Transport();_PENDING_TRANSPORTS.append(transport);r['pending_transports']=1
   record('batch_cast','224',lambda:transport.cast_into(dst,src),expected[:224],owners=(src,),view=dst,whole=owner,expected_whole=expected)
   transport.synchronize_and_release();assert not transport.pending;_PENDING_TRANSPORTS.remove(transport);r['pending_transports']=0;r['transport']=dict(transport.ledger[0])
   tables=(torch.arange(128,dtype=torch.int32).reshape(32,4),torch.arange(128,dtype=torch.int32).reshape(32,4));idx=torch.arange(32,dtype=torch.int64);idx[0]=-1;starts=torch.arange(33,dtype=torch.int32);positions=torch.arange(32,dtype=torch.int64)%4;sizes=torch.tensor([4,4],dtype=torch.int32);enabled=torch.tensor([True,False]);expected=torch.full((2,40),-91,dtype=torch.int64)
   reference_cpu(tables,idx,starts,positions,sizes,enabled,expected,32)
   ts=tuple(t.cuda()for t in tables);args=[t.cuda()for t in (idx,starts,positions,sizes,enabled)];out=torch.full((2,40),-91,dtype=torch.int64,device=device)
   record('slot_mapping','two_groups',lambda:compute_slot_mappings(ts,*args,out,32),expected[:,:32],owners=(*ts,*args),whole=out,expected_whole=expected)
   mapping=list(range(32));qs=list(range(33));computed=[53]*32;pos=[-91]*40;seq=[-92]*32;ep,es=reference_checked(mapping,qs,computed,pos,seq)
   inputs=[torch.tensor(x,dtype=dt,device=device)for x,dt in ((mapping,torch.int64),(qs,torch.int32),(computed,torch.int32))];pos_t=torch.tensor(pos,dtype=torch.int64,device=device);seq_t=torch.tensor(seq,dtype=torch.int32,device=device)
   record('pos_seq','32',lambda:prepare_pos_seq_lens(*inputs,pos_t,seq_t),torch.tensor(ep,dtype=torch.int64),owners=inputs,view=pos_t,whole=seq_t,expected_whole=torch.tensor(es,dtype=torch.int32))
   for dtype in ('int32','int64','float32'):
    out=torch.full((4,8),-91,dtype=getattr(torch,dtype),device=device);expected=out.clone();values=[1,-2,3,4];indices=[0,3];begin=[1,4];ends=[2,4];expected[0,1:3]=torch.tensor(values[:2],device=device);expected[3,4:6]=torch.tensor(values[2:],device=device);ledger=[]
    record('staged',dtype,lambda:staged_apply(out,indices,begin,values,ends,ledger=ledger),expected,view=out,whole=out,expected_whole=expected)
    assert indices==[0,3]and begin==[1,4]and values==[1,-2,3,4]and ends==[2,4]and ledger[0]['launch_attempts']==ledger[0]['launch_returns']==1
   from zero_kv_typed_lean_v1.registry import Allocation
   from zero_kv_phase2_typed_plan_v1.owned_view import chunk_view
   from zero_kv_phase2_typed_plan_v1.kernel import zero_words
   import cuda.tile as ct
   owner=torch.full((64,),-77,dtype=torch.int32,device=device);storage=owner.untyped_storage();allocation=Allocation(owner,storage,str(device),storage.data_ptr(),storage.nbytes());view=chunk_view(torch,allocation,64,192);expected=owner.clone();expected[16:48]=0
   def zero_leaf():ct.launch(torch.cuda.current_stream(device),(1,),zero_words,(view,));return None
   record('zero_words','32',zero_leaf,expected[16:48],view=view,whole=owner,expected_whole=expected)
  r['complete']=True;assert audit(r);save();return r
 except BaseException as exc:r['complete']=False;r['pending_transports']=len(_PENDING_TRANSPORTS);r['error']=type(exc).__name__+': '+str(exc);save();raise
