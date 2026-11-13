"""Four actual norm captures, no model integration or timing. CPU-default."""
import argparse,fcntl,hashlib,json,os,signal,sys,traceback
from pathlib import Path
from contextlib import contextmanager
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from norm_runtime_qualification_v1.compare import metrics

def descriptor(t):return dict(shape=list(t.shape),stride=list(t.stride()),storage_offset=t.storage_offset(),dtype=str(t.dtype),device=str(t.device))
def identity(fn):
 c=getattr(getattr(fn,'__func__',fn),'__code__',None)
 return dict(module=getattr(fn,'__module__',None),qualname=getattr(fn,'__qualname__',None),source=None if c is None else c.co_filename,line=None if c is None else c.co_firstlineno,source_sha256=None if c is None else sha(c.co_filename))
def logical(t,torch):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def aliases(values):
 out={}
 for a,x in values.items():
  for b,y in values.items():
   if a>=b:continue
   sx,sy=x.untyped_storage(),y.untyped_storage();xp,yp=sx.data_ptr(),sy.data_ptr()
   out[a+'|'+b]=dict(same_object=x is y,same_storage=xp==yp,storage_overlap=max(xp,yp)<min(xp+sx.nbytes(),yp+sy.nbytes()))
 return out
@contextmanager
def force_provider(op,provider):
 had='dispatch' in op.__dict__;prior=op.__dict__.get('dispatch');calls=[]
 def selected(*args,**kwargs):calls.append(1);return provider
 op.dispatch=selected
 try:yield calls
 finally:
  assert op.__dict__.get('dispatch') is selected,'Foreign dispatch change; not overwritten'
  if had:op.dispatch=prior
  else:delattr(op,'dispatch')
def result_gate(r):
 rows=r.get('records',[])
 complete=(r.get('complete') is True and r.get('gpu_executed') is True and len(rows)==4 and {(x['kind'],x['phase']) for x in rows}=={(k,p) for k in ('no_residual','with_residual') for p in ('prefill','decode')})
 contract=complete and all(x['contract_pass'] and all(x['inputs_preserved'].values()) for x in rows)
 raw=contract and all(c['raw_equal'] for x in rows for group in ('candidate_vs_capture','provider_vs_capture') for c in x[group])
 return dict(complete=complete,contract_pass=contract,raw_pass=raw,raw_difference_hold=contract and not raw)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True,type=Path);ap.add_argument('--manifest-sha',required=True);ap.add_argument('--output',required=True,type=Path);ap.add_argument('--run-gpu',action='store_true');a=ap.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha
 manifest=json.loads(a.manifest.read_text());assert manifest['schema']=='vllm-gemma-rms-boundary-v1' and manifest['complete'] and len(manifest['samples'])==4
 if not a.run_gpu:print(json.dumps(dict(executed=False,cases=4,manifest_sha=a.manifest_sha)));return 0
 lock='/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock';fd=int(os.environ['ALL_GEMM_LOCK_FD']);assert os.fstat(fd).st_ino==os.stat(lock).st_ino;fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 a.output.mkdir(exist_ok=False)
 for k,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/name).mkdir();os.environ[k]=str(a.output/name)
 r=dict(complete=False,gpu_executed=False,manifest_sha=a.manifest_sha,records=[],timing=False)
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 save();signal.alarm(175)
 try:
  import torch
  from vllm import ir
  from vllm.model_executor.layers.layernorm import GemmaRMSNorm
  from tileir_vllm_gemma_norm_v1 import gemma_rms_norm
  torch.set_num_threads(4)
  def load(item):
   assert sha(item['path'])==item['sha256'];t=torch.load(item['path'],map_location='cpu',weights_only=True)
   assert logical(t,torch)==item['logical_sha256'] and list(t.shape)==item['metadata']['shape']
   return t
  def restore(item):
   t=load(item);m=item['metadata'];shape=list(t.shape);stride=m['stride'];off=m['storage_offset']
   assert len(shape) in (1,2) and len(stride)==len(shape) and all(type(s) is int and s>0 for s in stride) and type(off) is int and off>=0
   assert stride[-1]==1 and (len(shape)==1 or stride[0]>=shape[1])
   n=off+sum((s-1)*st for s,st in zip(shape,stride))+1
   buf=torch.empty(n,dtype=t.dtype,device='cuda');v=torch.as_strided(buf,shape,stride,off);v.copy_(t);return v
  for sample in manifest['samples']:
   row=dict(kind=sample['kind'],phase=sample['phase']);r['records'].append(row);save()
   events=sample['provider_events'];assert len(events)==1,'Exact one captured provider required'
   event=events[0];name='rms_norm' if sample['kind']=='no_residual' else 'fused_add_rms_norm';assert event['op']==name
   op=getattr(ir.ops,name);provider=op.impls[event['provider']]
   assert provider.inplace==event['inplace'] and provider.supported==event['supported'] and provider.supported
   assert identity(provider.impl_fn)==event['implementation'] and identity(provider.func_impl_fn)==event['functional_wrapper']
   row['captured_provider']=event
   assert not any(sample['input_mutated'].values()),'Captured in-place mutation outside functional API scope'
   assert not any(x['storage_overlap'] for x in sample['pre_aliases'].values()),'Aliased input capture needs explicit reconstruction'
   reference=[load(x) for x in sample['outputs']];alloutputs={};row['inputs_preserved']={};row['output_metadata']={};row['aliases']={};row['return_kinds']={}
   for branch in ('provider','native_ir','candidate'):
    x=restore(sample['pre']['x']);r['gpu_executed']=True;save();w=restore(manifest['weights'][sample['weight_key']]);res=None if sample['kind']=='no_residual' else restore(sample['pre']['residual']);vals={'x':x,'weight':w}
    if res is not None:vals['residual']=res
    before={k:logical(t,torch) for k,t in vals.items()}
    with torch.inference_mode():
     if branch=='candidate':out=gemma_rms_norm(x,w,sample['epsilon'],res)
     elif branch=='native_ir':
      effective=w.float()+1.;fn=op.impls['native'].impl_fn
      out=fn(x,effective,sample['epsilon']) if res is None else fn(x,res,effective,sample['epsilon'])
     else:
      mod=GemmaRMSNorm(2560,eps=sample['epsilon']);mod.weight=torch.nn.Parameter(w,requires_grad=False)
      assert identity(mod.forward_cuda)==manifest['modules'][sample['kind']]['forward_cuda']
      with force_provider(op,provider) as calls:out=mod.forward_cuda(x,res)
      assert len(calls)==1,'Expected captured provider selection was not used'
      row['provider_forced_dispatch_calls']=len(calls)
    torch.cuda.synchronize();r['gpu_executed']=True
    outputs=list(out) if isinstance(out,tuple) else [out];assert len(outputs)==len(reference)
    row['return_kinds'][branch]='tuple' if isinstance(out,tuple) else 'tensor'
    row['inputs_preserved'][branch]=all(logical(t,torch)==before[k] for k,t in vals.items())
    row['output_metadata'][branch]=[descriptor(t) for t in outputs]
    row['aliases'][branch]=aliases(dict(vals,**{'output'+str(i):t for i,t in enumerate(outputs)}))
    alloutputs[branch]=[t.detach().cpu() for t in outputs]
    save()
   for branch in ('provider','native_ir','candidate'):row[branch+'_vs_capture']=[metrics(a,b) for a,b in zip(alloutputs[branch],reference)]
   row['candidate_vs_native_ir']=[metrics(a,b) for a,b in zip(alloutputs['candidate'],alloutputs['native_ir'])]
   row['contract_pass']=all(v==sample['return_kind'] for v in row['return_kinds'].values()) and all(row['aliases'][b]==sample['post_aliases'] for b in ('provider','candidate'))
   for branch in ('provider','candidate'):
    row['contract_pass'] &= all(all(g[k]==c['metadata'][k] for k in ('shape','stride','storage_offset','dtype')) for g,c in zip(row['output_metadata'][branch],sample['outputs']))
   save()
  r['complete']=True;verify_pins(R/'CPU-PINS.json');r['gate']=result_gate(r);save()
  return 0 if r['gate']['raw_pass'] else (2 if r['gate']['raw_difference_hold'] else 1)
 except BaseException as e:r.update(error=repr(e),traceback=traceback.format_exc());save();raise
 finally:signal.alarm(0)
if __name__=='__main__':raise SystemExit(main())
