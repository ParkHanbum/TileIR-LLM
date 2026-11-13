"""Two captured phases, derived selected-row table; no full checkpoint proof."""
import argparse,json,os,signal,sys
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent
sys.path[:0]=[str(P),str(R)]
from packed_grouped_qualification_v1.contract import sha,verify_pins

def check(m):
 assert m['schema']=='tp1-embedding-native-v1' and m['complete'] and m['observers_removed']
 assert m['counts']=={'prefill':1,'decode':15}
 assert len(m['samples'])==2 and {s['phase'] for s in m['samples']}=={'prefill','decode'}
 for s in m['samples']:
  n=1696 if s['phase']=='prefill' else 32
  assert s['input']['metadata']['shape']==[n] and s['input']['metadata']['stride']==[1] and s['input']['metadata']['storage_offset']==0
  assert s['input']['metadata']['dtype']=='torch.int64'
  w=s['weight_descriptor'];assert w['shape']==[248320,2560] and w['stride']==[2560,1] and w['storage_offset']==0 and w['dtype']=='torch.bfloat16'
  assert s['selected_rows_preserved'] and s['raw_rows_equal'] and s['finite']
 return True

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--output',required=True);p.add_argument('--run-gpu',action='store_true');a=p.parse_args()
 verify_pins(R/'CPU-PINS.json');assert sha(a.manifest)==a.manifest_sha
 m=json.loads(Path(a.manifest).read_text());check(m)
 if not a.run_gpu:print(json.dumps({'executed':False,'manifest_valid':True}));return 0
 assert 'ALL_GEMM_LOCK_FD' in os.environ
 os.fstat(int(os.environ['ALL_GEMM_LOCK_FD']))
 out=Path(a.output);out.mkdir(exist_ok=False)
 result=dict(complete=False,gpu_executed=True,manifest_sha=a.manifest_sha,records=[],scope='Derived zero table with captured live selected rows and original IDs. Not whole checkpoint equality; no timing.')
 def save():(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('175s worker')));signal.alarm(175)
 try:
  import torch
  import torch.nn.functional as F
  from tileir_vllm_tp1_embedding_v1 import embedding
  def digest(t):return __import__('hashlib').sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
  def load(e):
   assert sha(e['path'])==e['sha256']
   t=torch.load(e['path'],map_location='cpu',weights_only=True);assert digest(t)==e['logical_sha256'];return t
  def raw(a,b):return torch.equal(a.view(torch.int16),b.view(torch.int16))
  with torch.inference_mode():
   for s in m['samples']:
    ids,unique,inverse,rows,expected=[load(s[k]) for k in ('input','unique_ids','inverse','selected_weight_rows','output')]
    assert torch.equal(unique[inverse],ids) and raw(rows[inverse],expected)
    assert ids.min()>=0 and ids.max()<248320 and unique.unique().numel()==unique.numel()
    ids=ids.cuda();unique=unique.cuda();rows=rows.cuda();expected=expected.cuda()
    table=torch.zeros((248320,2560),dtype=torch.bfloat16,device='cuda');table.index_copy_(0,unique,rows)
    before=(digest(ids),digest(table));outputs=[];checks=[]
    for kind,fn in [('native',lambda:F.embedding(ids,table))]*2+[('tile',lambda:embedding(ids,table))]*2:
     y=fn();torch.cuda.synchronize()
     assert y.shape==expected.shape and y.dtype==torch.bfloat16 and y.stride()==(2560,1) and y.storage_offset()==0
     fresh=all(y.untyped_storage().data_ptr()!=t.untyped_storage().data_ptr() for t in [ids,table]+outputs)
     eq=raw(y,expected);finite=bool(torch.isfinite(y).all());strict=bool(torch.allclose(y.float(),expected.float(),atol=1e-5,rtol=1e-5))
     unchanged=before==(digest(ids),digest(table))
     checks.append(dict(kind=kind,raw_equal=eq,finite=finite,strict=strict,fresh=fresh,input_unchanged=unchanged));outputs.append(y)
     if not all((eq,finite,strict,fresh,unchanged)):
      result['records'].append(dict(phase=s['phase'],calls=checks,passed=False));save();return 2
    result['records'].append(dict(phase=s['phase'],calls=checks,raw_equal=True,finite=True,input_unchanged=True,reference_repeat_raw=raw(outputs[0],outputs[1]),candidate_repeat_raw=raw(outputs[2],outputs[3]),passed=True));save()
    del outputs,table,ids,unique,rows,expected,y
  result['complete']=True;return 0
 except BaseException as e:result['error']=repr(e);raise
 finally:signal.alarm(0);save()
if __name__=='__main__':raise SystemExit(main())
