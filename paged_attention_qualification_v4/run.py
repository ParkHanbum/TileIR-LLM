"""Captured FA2 paged attention replay. New logical TileIR candidate, raw equality unproven."""
from pathlib import Path
import json,hashlib,sys,os,signal,argparse
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P))
from packed_grouped_qualification_v1.contract import sha,verify_pins
from gdn_post_conv_qualification_v2.run import compare

def result_gate(r):
 rows=r.get('records',[]);complete=r.get('complete') is True and len(rows)==8
 return dict(complete=complete,raw_pass=complete and all(x['passed'] for x in rows),raw_difference_hold=r.get('raw_difference_hold') is True)
def validate_manifest(m):
 assert m['schema']=='attention-core-native-v1' and m['complete'] and len(m['calls'])==128 and len(m['samples'])==2
 for s in m['samples']:
  a=s['scalar_args'];assert a['fa_version']==2 and a['softmax_scale']==0.0625 and a['causal'] is True and a['num_splits']==0 and a['window_size']==[-1,-1] and not a['return_softmax_lse'] and not a['return_attn_probs']
  assert a['dropout_p']==0 and a['softcap']==0 and a['cp_world_size']==1
  for key in ('alibi_slopes','mask_mod','s_aux','dynamic_causal','q_v','output_scale','scheduler_metadata','aux_tensors'):assert a[key] is None
  assert s['inputs_preserved'] and s['finite_output']
 return m['samples']
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 assert sha(a.manifest)==a.manifest_sha;verify_pins(R/'CPU-PINS.json');samples=validate_manifest(json.loads(Path(a.manifest).read_text()))
 if not a.run_gpu:print('CPU manifest/pins PASS; no execution');return 0
 fd=int(os.environ['ALL_GEMM_LOCK_FD']);st=os.fstat(fd);ls=Path('/home/n/Documents/Codex/2026-10-31/task/tools/locks/fullgpu-native-model.lock').stat();assert(st.st_dev,st.st_ino)==(ls.st_dev,ls.st_ino)
 a.output.mkdir(exist_ok=False)
 for env,name in [('CUDA_TILE_CACHE_DIR','tilecache'),('CUDA_TILE_TEMP_DIR','tiletmp'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('TMPDIR','tmp')]:
  (a.output/name).mkdir();os.environ[env]=str(a.output/name)
 r=dict(complete=False,gpu_executed=True,manifest_sha=a.manifest_sha,records=[],scope='Two representative FA2 inputs, reconstructed sentinel-backed cache; no whole live owner/model/speed claim')
 def save():(a.output/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n')
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('285s')));signal.alarm(285);save()
 try:
  import torch
  from vllm.vllm_flash_attn import flash_attn_varlen_func as native
  from tileir_paged_attention_v4 import attention
  def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
  def load(e):
   assert sha(e['path'])==e['sha256'];t=torch.load(e['path'],map_location='cpu',weights_only=True);assert digest(t)==e['logical_sha256'];return t
  with torch.inference_mode():
   for s in samples:
    q=load(s['q']).cuda();target=load(s['output']).cuda();vk=load(s['valid_k']);vv=load(s['valid_v']);md={}
    for key,e in s['metadata'].items():
     t=load(e);d=e['metadata']
     if d['stride']==[0,0]:
      assert bool((t==1).all());md[key]=t[0,0].reshape(1).cuda().as_strided(d['shape'],d['stride'],d['storage_offset'])
     else:
      assert tuple(t.stride())==tuple(d['stride']) and d['storage_offset']==0;md[key]=t.cuda()
    kd,vd=s['tensors']['k'],s['tensors']['v'];shape=kd['shape'];stride=kd['stride'];ko=kd['storage_offset'];vo=vd['storage_offset'];assert vo==ko+256 and stride==vd['stride']
    extent=max(ko,vo)+sum((n-1)*st for n,st in zip(shape,stride))+1;assert extent*2<=768*1024*1024
    backing=torch.full((extent,),-37,dtype=torch.bfloat16,device='cuda');k=backing.as_strided(shape,stride,ko);v=backing.as_strided(shape,stride,vo)
    cursor=0;used=set();table=md['block_table'].cpu();lengths=md['seqused_k'].cpu().tolist()
    for b,L in enumerate(lengths):
     for pos in range(L):
      block=int(table[b,pos//shape[1]]);off=pos%shape[1];assert (block,off) not in used;used.add((block,off));k[block,off].copy_(vk[cursor+pos]);v[block,off].copy_(vv[cursor+pos])
     cursor+=L
    assert cursor==len(vk)==len(vv)
    inputs=dict(q=q,backing=backing,**md);before={n:digest(t) for n,t in inputs.items()};native_ref=None;tile_ref=None
    for kind in ('native','native','tile','tile'):
     out=torch.full_like(q,float('nan'))
     if kind=='native':ret=native(q=q,k=k,v=v,out=out,**md,**s['scalar_args'])
     else:ret=attention(q,k,v,md['cu_seqlens_q'],md['seqused_k'],md['block_table'],out)
     torch.cuda.synchronize();assert isinstance(ret,torch.Tensor)
     alias=ret.untyped_storage().data_ptr()==out.untyped_storage().data_ptr() and ret.shape==out.shape and ret.stride()==out.stride()
     metric=compare(target,out,torch);preserved=all(digest(t)==before[n] for n,t in inputs.items());ref=native_ref if kind=='native' else tile_ref
     repeat=True if ref is None else torch.equal(ref.view(torch.uint8),out.view(torch.uint8))
     passed=metric['raw_equal'] and metric['strict_failures']==0 and metric['candidate_nonfinite']==metric['reference_nonfinite']==0 and preserved and alias and repeat
     row=dict(phase=s['phase'],kind=kind,metrics=metric,inputs_preserved=preserved,return_out_alias=alias,repeat_raw=repeat,passed=passed);r['records'].append(row)
     if not passed:
      torch.save(out.cpu(),a.output/(s['phase']+'_'+kind+'_difference.pt'));r['raw_difference_hold']=True;save();return 2
     if kind=='native':native_ref=out.clone()
     else:tile_ref=out.clone()
     save()
    del inputs,backing,k,v,q,target,vk,vv,md,native_ref,tile_ref,out,ret
   r['complete']=True;return 0
 except BaseException as e:r['error']=repr(e);raise
 finally:signal.alarm(0);save()
if __name__=='__main__':raise SystemExit(main())
