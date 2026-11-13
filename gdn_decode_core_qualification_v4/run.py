"""Saved actual packed recurrent decode: native binding then candidate; no model run."""
import argparse,json,os,fcntl,signal,sys,traceback,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from norm_runtime_qualification_v2.compare import metrics as bf16_metrics
from gdn_decode_core_qualification_v4.storage_contract import untouched_ranges

def span(m):return m['storage_offset']+1+sum((n-1)*s for n,s in zip(m['shape'],m['stride']))
def result_gate(r):
 rows=r.get('records',[]);complete=r.get('complete')is True and len(rows)==2
 contract=complete and all(row.get('contract_pass')is True for row in rows)
 raw=contract and all(all(v['raw_equal']and v['strict_pass']and v['candidate_nonfinite']==v['reference_nonfinite']==0 for v in row['comparisons'].values())for row in rows)
 return dict(complete=complete,contract_pass=contract,raw_pass=raw,raw_difference_hold=contract and not raw)
def fp32_metrics(a,b,torch):
 a=a.detach().cpu().contiguous();b=b.detach().cpu().contiguous();assert a.dtype==b.dtype==torch.float32
 finite=torch.isfinite(a)&torch.isfinite(b);err=(a-b).abs();strict=(err<=1e-5+1e-5*b.abs())&finite
 ua=a.view(torch.int32).to(torch.int64)&0xffffffff;ub=b.view(torch.int32).to(torch.int64)&0xffffffff
 oa=torch.where(ua>=0x80000000,0xffffffff-ua,ua+0x80000000);ob=torch.where(ub>=0x80000000,0xffffffff-ub,ub+0x80000000)
 return dict(elements=a.numel(),raw_equal=bool(torch.equal(ua,ub)),raw_mismatches=int((ua!=ub).sum()),candidate_nonfinite=int((~torch.isfinite(a)).sum()),reference_nonfinite=int((~torch.isfinite(b)).sum()),strict_pass=bool(strict.all()),strict_failures=int((~strict).sum()),max_abs_finite=float(err[finite].max())if bool(finite.any())else None,max_fp32_ulp_finite=int((oa-ob).abs()[finite].max())if bool(finite.any())else None)
def merge(rows):
 r={}
 for k in rows[0]:
  vals=[x[k]for x in rows]
  r[k]=all(vals)if k in('raw_equal','strict_pass')else max(v for v in vals if v is not None)if k.startswith('max_')else sum(vals)
 return r
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha;m=json.loads(a.manifest.read_text());assert m['complete']and m['schema']=='vllm-gdn-packed-decode-native-v1';s,=m['samples']
 for p,h in m['source_pins'].items():assert sha(p)==h
 assert len(s['selected_slots'])==32 and not s['skip_positions']
 if not a.run_gpu:print(json.dumps(dict(gpu_executed=False,records=2,state_minspan_bytes=4*span(s['state_descriptor']))));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);lock='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock';assert os.fstat(fd).st_ino==os.stat(lock).st_ino;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for k,d in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TRITON_CACHE_DIR','triton'),('TMPDIR','tmp')]:
  (a.output/d).mkdir();os.environ[k]=str(a.output/d)
 r=dict(complete=False,gpu_executed=False,manifest_sha=a.manifest_sha,records=[],timing=False,scope='first layer0 ordinary decode actualbank,originalindices/pitches/offsets; unknown unselectedstate andholes synthetic sentinel notwholeowner')
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.alarm(285)
 try:
  import torch
  from vllm.third_party.flash_linear_attention.ops.fused_recurrent import fused_recurrent_gated_delta_rule_packed_decode as native
  from tileir_gdn_decode_core_v3 import packed_decode
  torch.set_num_threads(4);r['gpu_executed']=True
  def raw(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  def load(item):
   assert sha(item['path'])==item['sha256'];t=torch.load(item['path'],map_location='cpu',weights_only=True);assert raw(t)==item['logical_sha256'];return t
  cpu={k:load(item)for k,item in s['pre'].items()};before=load(s['selected_state_before']);after=load(s['selected_state_after']);reference=load(s['output']);slots=s['selected_slots'];sm=s['state_descriptor']
  assert span(sm)*4<=700*1024*1024
  def restore(value,d):
   backing=torch.full((span(d),),-7,dtype=value.dtype);view=torch.as_strided(backing,d['shape'],d['stride'],d['storage_offset']);view.copy_(value)
   gb=backing.to('cuda');return gb,torch.as_strided(gb,d['shape'],d['stride'],d['storage_offset']),raw(backing)
  for backend in ('native','tileir'):
   row=dict(backend=backend,comparisons={},contract_pass=False);r['records'].append(row);save()
   bufs={};inputs={};hashes={}
   for k,value in cpu.items():bufs[k],inputs[k],hashes[k]=restore(value,s['pre'][k]['metadata'])
   statecpu=torch.full((span(sm),),-7,dtype=torch.float32);sv=torch.as_strided(statecpu,sm['shape'],sm['stride'],sm['storage_offset']);sv[slots]=before
   sb=statecpu.to('cuda');state=torch.as_strided(sb,sm['shape'],sm['stride'],sm['storage_offset']);sv[slots]=after
   out=torch.full(reference.shape,-7,dtype=reference.dtype,device='cuda')
   assert tuple(out.stride())==tuple(s['output']['metadata']['stride']) and out.storage_offset()==s['output']['metadata']['storage_offset']
   fun=native if backend=='native'else packed_decode
   result=fun(**inputs,scale=s['arguments']['scale'],initial_state=state,out=out,use_qk_l2norm_in_kernel=s['arguments']['use_qk_l2norm_in_kernel']);torch.cuda.synchronize()
   row['tuple_identity']=isinstance(result,tuple)and len(result)==2 and result[0]is out and result[1]is state
   row['comparisons']['output']=bf16_metrics(out,reference)
   state_metrics=[fp32_metrics(state[slot],after[i],torch)for i,slot in enumerate(slots)];row['comparisons']['state']=merge(state_metrics);row['state_slot_metrics']=state_metrics
   postcpu=sb.detach().cpu()
   row['state_full_minspan_expected']=raw(postcpu)==raw(statecpu)
   intervals=untouched_ranges(sm,slots)
   row['untouched_minspan_preserved']=all(torch.equal(postcpu[lo:hi].view(torch.uint8),statecpu[lo:hi].view(torch.uint8))for lo,hi in intervals)
   row['untouched_elements_checked']=sum(hi-lo for lo,hi in intervals)
   artifacts={}
   saved_state=torch.empty_like(after)
   postview=torch.as_strided(postcpu,sm['shape'],sm['stride'],sm['storage_offset'])
   for i,slot in enumerate(slots):saved_state[i].copy_(postview[slot])
   for label,tensor in [('output',out.detach().cpu()),('selected_state',saved_state)]:
    path=a.output/(backend+'-'+label+'.pt');torch.save(tensor,path);artifacts[label]=dict(path=str(path),sha256=sha(path),logical_sha256=raw(tensor),shape=list(tensor.shape),dtype=str(tensor.dtype))
   row['artifacts']=artifacts
   del postview,saved_state,postcpu
   row['inputs_preserved']={k:raw(bufs[k])==hashes[k]for k in inputs}
   row['contract_pass']=row['tuple_identity']and row['untouched_minspan_preserved']and all(row['inputs_preserved'].values());save()
   assert row['contract_pass'],'state/alias/input contract HOLD'
   numeric=all(v['raw_equal']and v['strict_pass']and v['candidate_nonfinite']==v['reference_nonfinite']==0 for v in row['comparisons'].values())
   if backend=='native':assert numeric,'native binding HOLD; no candidate launch'
   row['numerical_pass']=numeric;save()
   del result,bufs,inputs,state,statecpu,sv,sb,out
  r['complete']=True;save()
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
 return 0 if result_gate(r)['raw_pass']else 2
if __name__=='__main__':raise SystemExit(main())
