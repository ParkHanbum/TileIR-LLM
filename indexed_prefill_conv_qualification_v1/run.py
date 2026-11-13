"""Frozen actual fixed prefill replay; CPU default, native2 then TileIR2, no timing."""
import argparse,json,os,fcntl,signal,sys,traceback,hashlib
from pathlib import Path
from types import SimpleNamespace
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from norm_runtime_qualification_v2.compare import metrics

def span(m):
 assert all(type(v)is int and v>0 for v in m['shape']+m['stride']) and m['storage_offset']>=0
 return m['storage_offset']+1+sum((n-1)*s for n,s in zip(m['shape'],m['stride']))
def result_gate(r):
 rows=r.get('records',[]);complete=r.get('complete')is True and r.get('gpu_executed')is True and len(rows)==4
 passed=complete and all(x.get('passed')is True for x in rows)
 return dict(complete=complete,raw_pass=passed,raw_difference_hold=r.get('raw_difference_hold')is True and not passed)
def walk_items(t):
 if isinstance(t,dict):
  if 'path'in t and 'sha256'in t:yield t
  else:
   for v in t.values():yield from walk_items(v)
 elif isinstance(t,list):
  for v in t:yield from walk_items(v)
def validate_manifest(m):
 assert m['complete'] and m['schema']=='indexed-causal-conv-native-v1'
 s,=[x for x in m['samples'] if x['phase']=='prefill']
 assert s['original_calls']==1 and s['state_unselected_unchanged'] and not s['skip_positions']
 assert s['arguments']==dict(bias=None,activation='silu',pad_slot_id=-1,null_block_id=0,block_idx_first_scheduled_token=None,block_idx_last_scheduled_token=None,initial_state_idx=None,num_computed_tokens=None,block_size_to_align=0,metadata={'type':'vllm.v1.attention.backends.gdn_attn.GDNAttentionMetadata'},validate_data=False)
 for item in walk_items(s):assert sha(item['path'])==item['sha256']
 for f,h in m['source_pins'].items():assert sha(f)==h
 return s

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha;s=validate_manifest(json.loads(a.manifest.read_text()))
 if not a.run_gpu:print(json.dumps(dict(gpu_executed=False,state_minspan_bytes=2*span(s['state_descriptor']),records=4,scope='one captured prefill native2/TileIR2, not full-model')));return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);held=os.fstat(fd);actual=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').stat();assert(held.st_dev,held.st_ino)==(actual.st_dev,actual.st_ino);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert a.output.is_absolute();a.output.mkdir(exist_ok=False)
 for env,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp'),('TRITON_CACHE_DIR','triton')]:
  (a.output/name).mkdir(exist_ok=True);os.environ[env]=str(a.output/name)
 r=dict(complete=False,gpu_executed=False,manifest_sha=a.manifest_sha,records=[],timing=False,scope='one saved layer0 ordinary prefill; selected state values restored; minspan holes/unselected cells sentinel, not unrelated original shared-owner bytes')
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('285s child bound')));signal.alarm(285)
 try:
  import torch
  from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
  from tileir_indexed_prefill_conv_v1 import causal_conv1d_prefill_indexed
  from norm_runtime_contract_capture_v1.observer import identity,metadata,aliases
  torch.set_num_threads(4);r['gpu_executed']=True;r['native_callable']=identity(causal_conv1d_fn);assert r['native_callable']['source_sha256']==s['helper']['source_sha256']
  def raw(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  def load(item):
   assert sha(item['path'])==item['sha256'];t=torch.load(item['path'],map_location='cpu',weights_only=True);assert raw(t)==item['logical_sha256'];return t
  vals={k:load(v) for k,v in s['pre'].items()};before=load(s['selected_state_before']);after=load(s['selected_state_after']);expected=load(s['output']);assert raw(vals['x'])==raw(load(s['post_x']))
  assert vals['query_start_loc'].tolist()==list(range(0,1697,53)) and not vals['has_initial_state'].any()
  slots=s['slots'];sm=s['state_descriptor'];assert span(sm)*2<=700*1024*1024
  def fresh(v,m):
   n=max(span(m),m.get('storage_nbytes',0)//v.element_size());cpu=torch.full((n,),False if v.dtype==torch.bool else -7,dtype=v.dtype)
   torch.as_strided(cpu,m['shape'],m['stride'],m['storage_offset']).copy_(v);gpu=cpu.to('cuda');return gpu,torch.as_strided(gpu,m['shape'],m['stride'],m['storage_offset']),raw(cpu)
  def restore_tree(node,memo):
   typ=node['type']
   if typ=='tensor':
    item=node['value'];md=item['metadata'];key=(md.get('device'),md.get('object_identity')) if 'object_identity'in md else item['path']
    if key not in memo:
     v=load(item)
     memo[key]=fresh(v,md)[1] if md['device'].startswith('cuda') else v.clone()
    return memo[key]
   if typ=='dict':return {restore_tree(k,memo):restore_tree(v,memo) for k,v in node['items']}
   if typ in ('list','tuple'):
    v=[restore_tree(x,memo) for x in node['items']];return tuple(v)if typ=='tuple'else v
   return node['value']
  def check_tree(value,node):
   typ=node['type']
   if typ=='tensor':return raw(value)==node['value']['logical_sha256']
   if typ=='dict':return len(value)==len(node['items'])and all(check_tree(value[restore_tree(k,{})],v)for k,v in node['items'])
   if typ in ('list','tuple'):return len(value)==len(node['items'])and all(check_tree(x,y)for x,y in zip(value,node['items']))
   return type(value).__name__==typ and value==node['value']
  previous={}
  for backend in ('native','tileir'):
   for repeat in range(2):
    row=dict(backend=backend,repeat=repeat,contract_pass=False,passed=False,comparisons={});r['records'].append(row);save()
    statecpu=torch.full((span(sm),),-7.0,dtype=torch.bfloat16);sv=torch.as_strided(statecpu,sm['shape'],sm['stride'],sm['storage_offset']);sv[slots]=before
    sb=statecpu.to('cuda');state=torch.as_strided(sb,sm['shape'],sm['stride'],sm['storage_offset'])
    restored={k:fresh(v,s['pre'][k]['metadata'])for k,v in vals.items()};inputs={k:v[1]for k,v in restored.items()}
    memo={};meta=SimpleNamespace(**{k:restore_tree(v,memo)for k,v in s['metadata_pre'].items()})
    if backend=='native':out=causal_conv1d_fn(**inputs,bias=None,conv_states=state,activation='silu',pad_slot_id=-1,null_block_id=0,block_size_to_align=0,metadata=meta,validate_data=False)
    else:out=causal_conv1d_prefill_indexed(inputs['x'],inputs['weight'],state,inputs['query_start_loc'],inputs['cache_indices'],inputs['has_initial_state'])
    torch.cuda.synchronize();sv[slots]=after
    row['comparisons']=dict(output=metrics(out,expected),selected_state=metrics(state[slots],after))
    row['state_backing_expected']=raw(sb)==raw(statecpu)
    row['inputs_backing_preserved']={k:raw(v[0])==v[2]for k,v in restored.items()}
    row['metadata_matches_expected']=all(check_tree(getattr(meta,k),v)for k,v in s['metadata_post'if backend=='native'else'metadata_pre'].items())
    row['output_descriptor']=metadata(out);row['aliases']=aliases(dict(inputs,state=state,output=out));om=s['output']['metadata']
    row['output_layout_exact']=all(row['output_descriptor'][k]==om[k]for k in ('shape','stride','storage_offset','dtype','storage_nbytes'))
    row['output_fresh']=not any(v['same_storage']or v['storage_overlap']or v['same_object']for k,v in row['aliases'].items()if'output'in k.split('|'))
    row['contract_pass']=row['state_backing_expected']and all(row['inputs_backing_preserved'].values())and row['metadata_matches_expected']and row['output_layout_exact']and row['output_fresh']
    row['repeat_raw_equal']=None if repeat==0 else raw(out)==previous[backend]
    row['passed']=row['contract_pass']and row['repeat_raw_equal']is not False and all(v['raw_equal']and v['strict_pass']and v['candidate_nonfinite']==v['reference_nonfinite']==0 for v in row['comparisons'].values())
    if not row['passed']:
     row['saved_outputs']={}
     for name,v in [('output',out),('selected_state',state[slots])]:
      f=a.output/(backend+str(repeat)+'_'+name+'.pt');torch.save(v.detach().cpu().contiguous(),f);row['saved_outputs'][name]=dict(path=str(f),sha256=sha(f))
     r.update(raw_difference_hold=True,halted_at=dict(backend=backend,repeat=repeat));save();return 2
    previous[backend]=raw(out);save();del out,state,sb,sv,statecpu,restored,inputs,meta,memo
  r['complete']=True;save();return 0
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
