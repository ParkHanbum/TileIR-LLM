"""Two actual attention gate cases; standalone inference replay, no performance claims."""
import argparse,fcntl,json,os,signal,sys,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from norm_runtime_qualification_v2.run import identity,logical,aliases,descriptor
from norm_runtime_qualification_v2.compare import metrics

def storage_elements(m):
 shape=m['shape'];stride=m['stride'];off=m['storage_offset']
 assert len(shape)==2 and len(stride)==len(shape)
 assert all(type(s) is int and s>0 for s in shape+stride) and type(off) is int and off>=0
 assert stride[-1]==1
 assert stride[0]>=shape[1]
 return off+sum((s-1)*st for s,st in zip(shape,stride))+1

def check_manifest(m):
 assert m['schema']=='vllm-attention-sigmoid-gate-v1' and m['complete'] is True
 assert len(m['samples'])==2 and {(s['kind'],s['phase']) for s in m['samples']}=={('attention_gate','prefill'),('attention_gate','decode')}
 assert m['weights']=={}
 for s in m['samples']:
  assert set(s['pre'])=={'x','gate'} and len(s['outputs'])==1 and s['return_kind']=='tensor'
  assert not any(s['input_mutated'].values()) and not any(v['storage_overlap'] for v in s['pre_aliases'].values())
  assert s['source_boundary']==m['modules']['gate'] and s['association']==['project_return_gate','attention_return','o_proj_input']
  for item in [*s['pre'].values(),*s['post_inputs'].values(),*s['outputs']]:
   d=item['metadata'];storage_elements(d)
   assert type(d['requires_grad']) is bool and type(d['is_leaf']) is bool and isinstance(d['python_type'],str)
 return m

def result_gate(r):
 rows=r.get('records',[])
 complete=r.get('complete') is True and r.get('gpu_executed') is True and len(rows)==2 and {x['phase'] for x in rows}=={'prefill','decode'}
 contract=complete and all(x.get('contract_pass') and all(x['inputs_preserved'].values()) for x in rows)
 raw=contract and all(c['raw_equal'] and c['candidate_nonfinite']==0 and c['reference_nonfinite']==0 for x in rows for c in x['comparisons'].values())
 return dict(complete=complete,contract_pass=contract,raw_pass=raw,raw_difference_hold=contract and not raw)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha
 manifest=check_manifest(json.loads(a.manifest.read_text()))
 for p,h in manifest['source_pins'].items():assert sha(p)==h
 if not a.run_gpu:print(json.dumps(dict(executed=False,cases=2)));return 0
 lock='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock';fd=int(os.environ['ALL_GEMM_LOCK_FD']);assert os.fstat(fd).st_ino==os.stat(lock).st_ino;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for k,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/name).mkdir();os.environ[k]=str(a.output/name)
 r=dict(complete=False,gpu_executed=False,manifest_sha=a.manifest_sha,records=[],timing=False,inference_only=True,provider_limits='Public Torch eager BF16 sigmoid materialization then multiply; not optimized Inductor fusion',original_descriptors_retained=True)
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.alarm(175)
 try:
  import torch
  from tileir_vllm_attention_gate_v1 import attention_sigmoid_gate
  torch.set_num_threads(4);r['replay_config']=dict(kind='Standalone public Torch eager BF16 sigmoid then multiply',not_model_execution=True)
  def load(item):
   assert sha(item['path'])==item['sha256'];t=torch.load(item['path'],map_location='cpu',weights_only=True)
   assert logical(t,torch)==item['logical_sha256'] and list(t.shape)==item['metadata']['shape'] and str(t.dtype)==item['metadata']['dtype'];return t
  def restore(item):
   t=load(item);m=item['metadata'];buf=torch.empty(storage_elements(m),dtype=t.dtype,device='cuda');v=torch.as_strided(buf,m['shape'],m['stride'],m['storage_offset']);v.copy_(t);return v
  meta=manifest['modules']['gate']
  for s in manifest['samples']:
   row=dict(kind='attention_gate',phase=s['phase'],inputs_preserved={},aliases={},output_metadata={},comparisons={},original_descriptors={k:v['metadata'] for k,v in s['pre'].items()});r['records'].append(row)
   reference_gpu=restore(s['outputs'][0]);reference=reference_gpu.cpu();row['restored_reference_descriptor']=descriptor(reference_gpu);r['gpu_executed']=True;outputs={}
   for branch in ('provider','candidate'):
    for repeat in (0,1):
     key=branch+str(repeat);x=restore(s['pre']['x']);g=restore(s['pre']['gate']);r['gpu_executed']=True;save()
     vals=dict(x=x,gate=g);before={k:logical(v,torch) for k,v in vals.items()}
     with torch.inference_mode():
      if branch=='candidate':out=attention_sigmoid_gate(x.detach(),g.detach())
      else:
       sigmoid=torch.sigmoid(g);assert sigmoid.dtype==torch.bfloat16
       out=x*sigmoid
     assert isinstance(out,torch.Tensor);torch.cuda.synchronize()
     row['inputs_preserved'][key]=all(logical(v,torch)==before[k] for k,v in vals.items())
     row['aliases'][key]=aliases(dict(vals,output=out));row['output_metadata'][key]=descriptor(out)
     outputs[key]=out.detach().cpu();row['comparisons'][key+'_vs_capture']=metrics(outputs[key],reference);save()
   for branch in ('provider','candidate'):row['comparisons'][branch+'_repeat']=metrics(outputs[branch+'0'],outputs[branch+'1'])
   row['comparisons']['candidate_vs_provider']=metrics(outputs['candidate0'],outputs['provider0'])
   row['contract_pass']=all(v==s['post_aliases'] for v in row['aliases'].values()) and all(all(d[k]==s['outputs'][0]['metadata'][k] for k in ('shape','stride','storage_offset','dtype')) for d in row['output_metadata'].values());save()
  verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha;r['complete']=True;r['gate']=result_gate(r);save()
  return 0 if r['gate']['raw_pass'] else 2 if r['gate']['raw_difference_hold'] else 1
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
