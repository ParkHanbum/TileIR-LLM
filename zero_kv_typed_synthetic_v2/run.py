import argparse,os,sys,json,hashlib,signal
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from zero_kv_typed_synthetic_v2.cases import CASES,intervals
from zero_kv_typed_synthetic_v2.gates import qualification,audit
from zero_kv_phase2_typed_plan_v1.geometry import merge
PYTHON='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
LOCK='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock'
def pins():
 for p,h in json.loads((R/'PINS.json').read_text()).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-gpu',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
 if not a.run_gpu:print(json.dumps(dict(executed=False,private_cases=4,native_kernel_calls=3,tile_api_calls=4,framework_imported=False)));return 0
 assert str(Path(sys.executable).absolute())==PYTHON and os.environ.get('VLLM_COMPARISON_PARENT_APPROVED')=='1'
 fd=int(os.environ['GEMM_COMPOSED_LOCK_FD']);s=os.fstat(fd);t=os.stat(LOCK);assert(s.st_dev,s.st_ino)==(t.st_dev,t.st_ino)
 from all_gemm_sixteen_sampler_composition_v2.cache_gate import preflight
 qualification(R);pins();cache=preflight(R);assert a.output and not a.output.exists();a.output.mkdir()
 r=dict(complete=False,cache_preseed=cache,records=[],scope='private synthetic; not model; no timing',candidate_compile_status='not_attempted');code=1
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.alarm(300)
 try:
  import torch,cuda.tile as ct
  from vllm.v1.worker.utils import _zero_kv_blocks_kernel
  from zero_kv_registry_phase1_v4.registry import Allocation
  from zero_kv_typed_synthetic_v2.api import zero_owned_intervals
  assert torch.__version__=='2.13.0+cu130'
  flags=lambda:dict(bf16=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,split_k=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)
  r['precision_initial']=flags();assert r['precision_initial']==dict(bf16=True,split_k=True)
  torch.set_num_threads(4)
  for c in CASES:
   # Reproducible random bits, including NaN/Inf/signed-zero bit patterns.
   gen=torch.Generator(device='cpu').manual_seed(20261113)
   pre=torch.randint(-2147483648,2147483647,(c['words'],),dtype=torch.int32,generator=gen)
   special=[0,-2147483648,2139095040,-8388608,2143289345,2139095041,-4194303]
   for j,v in enumerate(special):pre[j]=v
   spans=merge(c['words']*4,intervals(c));native_small=None;native_hashes=None
   for arm in ('native','tile'):
    owner=pre.to('cuda');storage=owner.untyped_storage();allocation=Allocation(owner,storage,str(owner.device),storage.data_ptr(),storage.nbytes())
    stream=torch.cuda.current_stream(owner.device);tables=[];table_cpus=[]
    ledger=dict(native_attempted=0,native_returned=0,tile=dict(attempted=0,returned=0))
    r['current']=dict(case=c['name'],arm=arm,source_calls=ledger);save()
    if arm=='native' and c['ids']:
     values=([owner.data_ptr()+o*4 for o,s,p,b in c['segments']],[s for o,s,p,b in c['segments']],[p for o,s,p,b in c['segments']],c['ids'])
     for vals,dtype in zip(values,(torch.uint64,torch.int64,torch.int64,torch.int64)):
      cpu=torch.tensor(vals,dtype=dtype);table_cpus.append(cpu);tables.append(cpu.to('cuda'))
     page=max(x[2]for x in c['segments']);block=min(1<<(page-1).bit_length(),1024)
     ledger['native_attempted']+=1;save()
     _zero_kv_blocks_kernel[(len(c['ids']),len(c['segments']),(page+block-1)//block)](*tables,BLOCK_SIZE=block)
     ledger['native_returned']+=1;save()
    elif arm=='tile':
     r['candidate_compile_status']='attempting';save();launch=zero_owned_intervals(allocation,intervals(c),torch=torch,ct=ct,ledger=ledger['tile']);r['candidate_compile_status']='completed'
    torch.cuda.synchronize();same=torch.cuda.current_stream(owner.device).cuda_stream==stream.cuda_stream
    readonly=all(torch.equal(t.cpu().view(torch.uint8),v.view(torch.uint8))for t,v in zip(tables,table_cpus))
    hashes=[];raw=True;small=[];chunkwords=2*1024**2 #8MiB transfer/expected chunk, independent of64MiB launch chunks.
    for lo in range(0,c['words'],chunkwords):
     hi=min(c['words'],lo+chunkwords);actual=owner[lo:hi].cpu();expected=pre[lo:hi].clone()
     for x,y in spans:
      left=max(lo,x//4);right=min(hi,y//4)
      if left<right:expected[left-lo:right-lo]=0
     raw=raw and torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
     hashes.append(hashlib.sha256(memoryview(actual.numpy()).cast('B')).hexdigest())
     if c['words']<4096:small.append(actual.clone())
    if arm=='native':native_hashes=hashes;native_small=torch.cat(small)if small else None;cross=True
    else:cross=(torch.equal(torch.cat(small).view(torch.uint8),native_small.view(torch.uint8))if small else hashes==native_hashes)
    r['records'].append(dict(case=c['name'],arm=arm,expected_full_raw=raw,readonly_tables_raw=readonly,same_current_stream=same,cross_arm_equal=cross,cross_arm_method='direct_raw'if small else 'all8MiB_chunk_SHA256',all_output_bytes=c['words']*4,target_bytes=sum(y-x for x,y in spans),candidate_launches=launch['launches']if arm=='tile'else 0,source_calls=ledger))
    save()
    if not(raw and readonly and same and cross):r['hold']=True;save();return 2
    del owner,allocation,storage,tables,table_cpus,actual,expected,small
    if arm=='native' and c['ids']:del cpu
    # Loop table vars retain final tensors after comprehension construction.
    if arm=='native' and c['ids']:del vals,values
   del pre,native_small
  pins();r['precision_final']=flags();assert r['precision_final']==r['precision_initial'];r['complete']=True;save();code=0
 except BaseException:
  r['hold']=True;r['error_code']='compile_or_runtime_or_contract_failure';save();code=2
 finally:
  try:
   import torch
   torch.cuda.synchronize();r['synchronized']=True
  except BaseException:r['synchronized']=False;code=1
  signal.alarm(0);save()
  if code==0 and not audit(r)['passed']:r['complete']=False;r['hold']=True;save();code=2
 return code
if __name__=='__main__':raise SystemExit(main())
