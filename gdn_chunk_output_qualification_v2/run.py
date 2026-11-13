"""Stage6 only; actual native Ai; native2 then candidate2, raw fail-closed."""
import argparse,hashlib,json,os,signal,sys,importlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from gdn_post_conv_qualification_v2.run import compare
C=P/'gdn_prefill_stage_capture_v1';B=C/'gdnStagesCapture01'
def result_gate(r):
 rows=r.get('records',[]);complete=r.get('complete') is True and len(rows)==4
 raw=complete and [(x['arm'],x['repeat']) for x in rows]==[('native',0),('native',1),('tile',0),('tile',1)] and all(x.get('passed') is True for x in rows) and r.get('initial_flags')==r.get('final_flags')==[True,True]
 return dict(complete=complete,raw_pass=raw,raw_difference_hold=r.get('raw_difference_hold') is True and not raw)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 assert sha(a.manifest)==a.manifest_sha;verify_pins(R/'CPU-PINS.json')
 m=json.loads(Path(a.manifest).read_text());s,=m['samples'];cap=json.loads((B/'RESULT.json').read_text())
 assert cap['complete'] and cap['artifacts_written'] and cap['full_final_raw'] and cap['existing_A_raw_equal'] and cap['actual_solver_precision']=='ieee' and cap['actual_solver_USE_TMA'] is False
 items={**{n:cap['artifacts'][src] for n,src in [('h','h'),('v','v_new'),('g','g_cumsum'),('expected_o','o')]},**{n:s['pre'][n] for n in ('q','k','cu_seqlens','chunk_indices')}}
 for e in items.values():assert sha(e['path'])==e['sha256']
 if not a.run_gpu:print('CPU source/input provenance PASS; no execution');return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.fstat(fd);ls=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').stat();assert(st.st_dev,st.st_ino)==(ls.st_dev,ls.st_ino)
 a.output.mkdir(exist_ok=False)
 for env,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/name).mkdir();os.environ[env]=str(a.output/name)
 r=dict(complete=False,gpu_executed=False,manifest_sha=a.manifest_sha,records=[],timing=False,scope='layer0 native stage6 input, not KKT adoption or full-core/model equivalence',capture_result_sha=sha(B/'RESULT.json'))
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('585s')));signal.alarm(585)
 try:
  from gdn_prefill_stage_capture_v1.cache_groups import relocate
  cp=json.loads((C/'CACHE-PINS.json').read_text());srcroot=Path(cp['source'])
  for rel,h in cp['files'].items():
   src=srcroot/rel;assert not src.is_symlink() and sha(src)==h;dst=a.output/'triton'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(dst)==h
  for p in sorted((a.output/'triton').rglob('__grp__*.json')):
   names=[f.name for f in p.parent.iterdir() if f.is_file() and f!=p];p.write_text(json.dumps(relocate(json.loads(p.read_text()),p.parent,names)))
  import torch
  from tileir_gdn_chunk_output_v2 import chunk_output as tile
  from tileir_gdn_chunk_cumsum_v1.contract import validate_metadata
  assert os.environ.get('FLA_USE_FAST_OPS','0')=='0'
  native=importlib.import_module('vllm.third_party.flash_linear_attention.ops.chunk_o')
  r.update(gpu_executed=True)
  flags=lambda:[torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k]
  r['initial_flags']=flags();assert r['initial_flags']==[True,True];torch.set_num_threads(4)
  def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  cpus={}
  for key,e in items.items():
   t=torch.load(e['path'],map_location='cpu',weights_only=True);assert digest(t)==e['logical_sha256'];d=e.get('metadata',e)
   assert list(t.shape)==d['shape'] and list(t.stride())==d['stride'] and str(t.dtype)==d['dtype'] and t.storage_offset()==0 and t.is_contiguous();cpus[key]=t
  assert all(torch.isfinite(t).all() for t in cpus.values())
  validate_metadata(cpus['cu_seqlens'].tolist(),cpus['chunk_indices'].tolist())
  inputs={k:v.cuda() for k,v in cpus.items() if not k.startswith('expected_')};previous={}
  with torch.inference_mode():
   for arm in ('native','tile'):
    for repeat in range(2):
     row=dict(arm=arm,repeat=repeat,passed=False);r['records'].append(row);save()
     out=native.chunk_fwd_o(**inputs,scale=128**-0.5,chunk_size=64,core_attn_out=None) if arm=='native' else tile(**inputs)
     outputs=(out,)
     torch.cuda.synchronize();assert isinstance(outputs,tuple) and len(outputs)==1
     row['comparisons']={};row['storage_contract']=True;row['repeat_raw_equal']=True
     cpupair=[]
     for name,out in zip(('o',),outputs):
      expected=cpus['expected_'+name];assert out.is_cuda and out.dtype==expected.dtype and tuple(out.shape)==tuple(expected.shape) and out.stride()==expected.stride() and out.storage_offset()==0 and out.untyped_storage().nbytes()==expected.numel()*expected.element_size()
      row['storage_contract'] &= all(out.untyped_storage().data_ptr()!=v.untyped_storage().data_ptr() for v in inputs.values())
      cpu=out.cpu();cpupair.append(cpu);row['comparisons'][name]=compare(cpus['expected_'+name],cpu,torch)
      row['repeat_raw_equal'] &= arm not in previous or torch.equal(previous[arm][len(cpupair)-1].view(torch.uint8),cpu.view(torch.uint8))
      torch.save(cpu,a.output/(arm+str(repeat)+'-'+name+'.pt'))
     row['storage_contract'] &= len({o.untyped_storage().data_ptr() for o in outputs})==1
     row['inputs_preserved']={k:digest(v)==items[k]['logical_sha256'] for k,v in inputs.items()}
     row['passed']=all(c['raw_equal'] and c['strict_failures']==c['reference_nonfinite']==c['candidate_nonfinite']==0 for c in row['comparisons'].values()) and row['storage_contract'] and all(row['inputs_preserved'].values()) and row['repeat_raw_equal']
     save()
     if not row['passed']:r['raw_difference_hold']=True;return 2
     previous[arm]=cpupair
  r['final_flags']=flags();assert r['final_flags']==r['initial_flags'];r['complete']=True;verify_pins(R/'CPU-PINS.json');return 0
 except BaseException as e:r['error']=repr(e);raise
 finally:signal.alarm(0);save()
if __name__=='__main__':raise SystemExit(main())
