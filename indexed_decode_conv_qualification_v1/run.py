"""Actual saved indexed decode; no whole-model run or performance measurement."""
import argparse,json,os,fcntl,signal,sys,traceback,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from norm_runtime_qualification_v2.compare import metrics

def span(m):
 assert all(type(v) is int and v>0 for v in m['shape']+m['stride']) and m['storage_offset']>=0
 return m['storage_offset']+1+sum((n-1)*s for n,s in zip(m['shape'],m['stride']))
def result_gate(r):
 rows=r.get('records',[]);complete=r.get('complete') is True and len(rows)==4
 contract=complete and all(x.get('contract_pass') is True for x in rows)
 raw=contract and all(all(v['raw_equal'] and v['strict_pass'] and v['candidate_nonfinite']==v['reference_nonfinite']==0 for v in x['comparisons'].values()) for x in rows)
 return dict(complete=complete,contract_pass=contract,raw_pass=raw,raw_difference_hold=contract and not raw)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha
 m=json.loads(a.manifest.read_text());assert m['complete'] and m['schema']=='indexed-causal-conv-native-v1'
 s,=[x for x in m['samples'] if x['phase']=='decode']
 assert s['arguments']['out'] is None and s['arguments']['bias'] is None and s['arguments']['activation']=='silu' and not s['skip_positions']
 for p,h in m['source_pins'].items():assert sha(p)==h
 if not a.run_gpu:print(json.dumps(dict(gpu_executed=False,state_minspan_bytes=2*span(s['state_descriptor']),selected_slots=len(s['slots']),records=4)));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);lock='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock';assert os.fstat(fd).st_ino==os.stat(lock).st_ino;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for k,d in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/d).mkdir();os.environ[k]=str(a.output/d)
 r=dict(complete=False,gpu_executed=False,manifest_sha=a.manifest_sha,records=[],timing=False,scope='saved layer0 firstdecode plus synthetic null control; unselected logical cells and minspan holes filled sentinel, not original other-owner bytes')
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.alarm(285)
 try:
  import torch
  from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
  from tileir_indexed_decode_conv_v1 import causal_conv1d_update_indexed
  torch.set_num_threads(4);r['gpu_executed']=True
  def raw(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  def load(item):
   assert sha(item['path'])==item['sha256'];t=torch.load(item['path'],map_location='cpu',weights_only=True);assert raw(t)==item['logical_sha256'];return t
  x0=load(s['pre']['x']);w0=load(s['pre']['weight']);i0=load(s['pre']['conv_state_indices']);before=load(s['selected_state_before']);after=load(s['selected_state_after']);y0=load(s['output']);assert raw(y0)==raw(load(s['post_x']))
  slots=s['slots'];sm=s['state_descriptor'];xm=s['pre']['x']['metadata'];wm=s['pre']['weight']['metadata'];im=s['pre']['conv_state_indices']['metadata']
  assert span(sm)*2<=700*1024*1024
  def fresh(vals,meta):
   cpu=torch.full((span(meta),),-7.0,dtype=vals.dtype);view=torch.as_strided(cpu,meta['shape'],meta['stride'],meta['storage_offset']);view.copy_(vals)
   gpu=cpu.to('cuda');return gpu,torch.as_strided(gpu,meta['shape'],meta['stride'],meta['storage_offset']),cpu
  for nullcase in (False,True):
   for backend in ('native','tileir'):
    row=dict(backend=backend,synthetic_null=nullcase,comparisons={},contract_pass=False);r['records'].append(row);save()
    statecpu=torch.full((span(sm),),-7.0,dtype=torch.bfloat16);sv=torch.as_strided(statecpu,sm['shape'],sm['stride'],sm['storage_offset'])
    sv[slots]=before
    statebuf=statecpu.to('cuda');state=torch.as_strided(statebuf,sm['shape'],sm['stride'],sm['storage_offset'])
    xb,x,xexpect=fresh(x0,xm);wb,w,wexpect=fresh(w0,wm);ivals=i0.clone()
    if nullcase:ivals[0]=0
    ib,indices,iexpect=fresh(ivals,im)
    expected=y0.clone();expected_state=after.clone()
    if nullcase:
     expected[0]=x0[0];expected_state[slots.index(int(i0[0]))]=before[slots.index(int(i0[0]))]
    sv[slots]=expected_state
    torch.as_strided(xexpect,xm['shape'],xm['stride'],xm['storage_offset']).copy_(expected)
    if backend=='native':y=causal_conv1d_update(x,state,w,None,'silu',conv_state_indices=indices,validate_data=False)
    else:y=causal_conv1d_update_indexed(x,state,w,indices)
    torch.cuda.synchronize()
    row['comparisons']['output']=metrics(y,expected);row['comparisons']['selected_state']=metrics(state[slots],expected_state)
    row['alias']=dict(same_object=y is x,same_storage=y.untyped_storage().data_ptr()==x.untyped_storage().data_ptr(),shape=list(y.shape),stride=list(y.stride()),offset=y.storage_offset())
    # Full minspan bytes checks include every hole and unselected logical slot.
    row['state_backing_expected']=raw(statebuf)==raw(statecpu)
    row['x_backing_expected']=raw(xb)==raw(xexpect)
    row['weight_preserved']=raw(wb)==raw(wexpect);row['indices_preserved']=raw(ib)==raw(iexpect)
    row['contract_pass']=all(row[k] for k in ('state_backing_expected','x_backing_expected','weight_preserved','indices_preserved')) and row['alias']['same_storage'] and not row['alias']['same_object'] and tuple(y.stride())==tuple(x.stride()) and y.storage_offset()==x.storage_offset()
    save()
    assert row['contract_pass'],'mutation/alias/holes contract HOLD'
    assert all(v['raw_equal'] and v['strict_pass'] and v['candidate_nonfinite']==v['reference_nonfinite']==0 for v in row['comparisons'].values()),'numerical HOLD; stop remaining launches'
    del statebuf,state,statecpu,sv,xb,x,wb,w,ib,indices,y
  r['complete']=True;save()
 except BaseException as e:
  r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
 return 0 if result_gate(r)['raw_pass'] else 2
if __name__=='__main__':raise SystemExit(main())
