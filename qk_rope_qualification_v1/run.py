"""Same actual packed inputs, preserved shared storage and strided MRoPE positions."""
import argparse,fcntl,json,os,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from norm_runtime_qualification_v2.compare import metrics
MANIFEST=P/'qk_rope_runtime_capture_v1/qkRopeCapture01/bank/MANIFEST.json'
MANIFEST_SHA='1953f2a5cc9cfae4f34197cb84e65a85180839a6d62c725cc995d4ce070e06df'
def result_gate(r):
 rows=r.get('records',[]);complete=r.get('complete') is True and len(rows)==8 and {(x['phase'],x['branch'],x['repeat']) for x in rows}=={(p,b,i) for p in ('prefill','decode') for b in ('native','tileir') for i in (0,1)}
 contract=complete and all(x['contract_pass'] for x in rows);raw=contract and all(all(v['raw_equal'] and v['candidate_nonfinite']==0 and v['reference_nonfinite']==0 for v in x['comparisons'].values()) and x['repeat_equal'] for x in rows)
 return dict(complete=complete,contract_pass=contract,raw_pass=raw,raw_difference_hold=any(not v['raw_equal'] for x in rows for v in x.get('comparisons',{}).values()))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-gpu',action='store_true');ap.add_argument('--output',required=True);a=ap.parse_args();verify_pins(R/'CPU-PINS.json');assert sha(MANIFEST)==MANIFEST_SHA
 if not a.run_gpu:print(json.dumps(dict(executed=False,cases=2,repeats=2,arms=['native','tileir'],timing=False)));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);assert os.fstat(fd).st_ino==os.stat('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').st_ino;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 out=Path(a.output);out.mkdir(exist_ok=False)
 for k,n in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TRITON_CACHE_DIR','triton'),('TMPDIR','tmp')]:
  (out/n).mkdir();os.environ[k]=str(out/n)
 r=dict(complete=False,gpu_executed=False,manifest_sha=MANIFEST_SHA,records=[],timing=False,scope='Captured layer3 same-input native/TileIR functional contract only')
 def save():(out/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.alarm(285)
 try:
  import torch
  from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate as native
  from tileir_vllm_qk_rope_v1 import fused_qk_rmsnorm_rope_gate as tileir
  from norm_runtime_contract_capture_v1.observer import aliases
  torch.set_num_threads(4);m=json.loads(MANIFEST.read_text());assert m['complete'] and len(m['samples'])==2
  for s in m['samples']:
   phase=s['phase'];refs=[torch.load(t['path'],weights_only=True,map_location='cpu') for t in s['outputs']]
   for t in [*s['pre'].values(),*s['outputs']]:
    assert sha(t['path'])==t['sha256'] and sha(t['storage_path'])==t['storage_sha256']
   prior={}
   for branch,fn in [('native',native),('tileir',tileir)]:
    for repeat in (0,1):
     buffers={};vals={}
     for name,t in s['pre'].items():
      d=t['metadata'];key=(d['storage_ptr'],d['storage_bytes'])
      if key not in buffers:buffers[key]=torch.load(t['storage_path'],map_location='cpu',weights_only=True).to('cuda')
      raw=buffers[key];dtype=getattr(torch,d['dtype'].split('.')[-1]);vals[name]=torch.empty(0,dtype=dtype,device='cuda').set_(raw.untyped_storage(),d['storage_offset'],d['shape'],d['stride'])
     assert vals['q_gate'].untyped_storage().data_ptr()==vals['k'].untyped_storage().data_ptr()==vals['qkv'].untyped_storage().data_ptr()
     assert tuple(vals['positions'].stride())==(2049,1) and vals['k'].storage_offset()==8192
     pos=vals['positions'].cpu();assert bool(((pos>=0)&(pos<262144)).all())
     import hashlib
     def digest(t):return hashlib.sha256(t.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
     before={key:digest(raw) for key,raw in buffers.items()};r['gpu_executed']=True
     prm=s['parameters'];inputs=[vals[k] for k in ('q_gate','k','q_weight','k_weight','cache','positions')]
     with torch.inference_mode():ys=fn(*inputs,prm['eps'],prm['num_q_heads'],prm['num_kv_heads'],prm['head_dim'],prm['rotary_dim'],mrope_section=prm['mrope_section'],norm_beta=prm['norm_beta'])
     torch.cuda.synchronize();assert isinstance(ys,tuple) and len(ys)==3
     aa=aliases(dict(vals,**dict(zip(('q_out','k_out','gate_out'),ys))))
     fresh=all(not v['storage_overlap'] for key,v in aa.items() if any(n in key.split('|') for n in ('q_out','k_out','gate_out')))
     preserved=all(digest(raw)==before[key] for key,raw in buffers.items());layout=all(y.dtype==torch.bfloat16 and y.is_contiguous() and y.storage_offset()==0 and tuple(y.shape)==tuple(ref.shape) for y,ref in zip(ys,refs))
     cp=[y.cpu() for y in ys];comparisons={name:metrics(y,ref) for name,y,ref in zip(('q','k','gate'),cp,refs)};hashes=[digest(x) for x in cp];repeat_equal=branch not in prior or hashes==prior[branch];prior[branch]=hashes
     expected_gate=vals['q_gate'].reshape(-1,16,512)[:,:,256:].reshape(-1,4096);gatecopy=digest(ys[2])==digest(expected_gate)
     row=dict(phase=phase,branch=branch,repeat=repeat,comparisons=comparisons,inputs_preserved=preserved,fresh_disjoint=fresh,layout_pass=layout,gate_copy_raw=gatecopy,repeat_equal=repeat_equal,contract_pass=preserved and fresh and layout and gatecopy,original_descriptors={k:v['metadata'] for k,v in s['pre'].items()},saved_outputs={});r['records'].append(row)
     if repeat==0:
      for name,y in zip(('q','k','gate'),cp):
       f=out/(phase+'_'+branch+'_'+name+'.pt');torch.save(y,f);row['saved_outputs'][name]=dict(path=str(f),sha256=sha(f))
     save();assert row['contract_pass'],'Input/alias/layout/gate-copy contract failed'
     if not all(c['raw_equal'] and c['candidate_nonfinite']==0 and c['reference_nonfinite']==0 for c in comparisons.values()) or not repeat_equal:r['gate']=result_gate(r);save();return 2
  r['complete']=True;r['gate']=result_gate(r);verify_pins(R/'CPU-PINS.json');save();return 0 if r['gate']['raw_pass'] else 2
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
