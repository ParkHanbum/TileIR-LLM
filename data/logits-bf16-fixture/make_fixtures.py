"""Independent CPU-only deterministic synthetic stored-BF16 fixtures and oracles."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;D=R/'fixtures';D.mkdir(exist_ok=False)
M=32;N=248320;T=1024;B=243;E=M*N

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,a):
 p=D/name
 with p.open('xb')as f:f.write(a.tobytes(order='C'))
 return dict(file=str(p),bytes=p.stat().st_size,sha256=sha(p))
i=np.arange(E,dtype=np.uint32);u=i*np.uint32(1664525)+np.uint32(1013904223);u^=u>>np.uint32(16)
nom=((u&np.uint32(0x8000))|(np.uint32(0x3b80)+(u>>np.uint32(16))%np.uint32(1408))).astype('<u2').reshape(M,N)
allbits=np.arange(65536,dtype=np.uint16);finitebits=allbits[(allbits&0x7f80)!=0x7f80];assert len(finitebits)==65280
adv=np.resize(finitebits,E).reshape(M,N).astype('<u2');adv[1]=np.where(np.arange(N)%2,0x8000,0).astype('<u2')
adv[2].fill(0xbf80);adv[2,-1]=0xbf00
adv[3].fill(0xc000)
for x in [0,1023,1024,247807,247808,248319]:adv[3,x]=0x40c0
adv[4].fill(0xc000);adv[4,-1]=0xbf80
adv[5].fill(0xc040)
for x in [1023,1024,247807,247808,248319]:adv[5,x]=0x4090
indices=[0,1023,1024,N-1,N,31*N+247808,E-1]
cases=[('adversarial',adv,True),('nominal',nom,True)]
for name,patterns in [('nan',[0x7f81,0xff81,0x7fc1,0xffc1,0x7fff,0xffff,0x7fa5]),('posinf',[0x7f80]*7),('neginf',[0xff80]*7)]:
 a=nom.copy();a.reshape(-1)[indices]=patterns;cases.append((name,a,False))
rows=[]
for name,a,ok in cases:
 finite=(a.reshape(-1)&0x7f80)!=0x7f80;status=np.all(finite.reshape(-1,T),axis=1).astype('<i4');assert len(status)==7760 and bool(np.all(status==1))==ok
 row=dict(id=name,synthetic=True,expected_finite=ok,input=put(name+'.bf16',a),expected_status=put(name+'.status.i32',status),shape=[M,N],strides=[N,1],dtype='bf16',input_bytes=E*2)
 if ok:
  f=(a.astype(np.uint32)<<16).view('<f4');v=np.empty((M,B),dtype='<f4');ix=np.empty((M,B),dtype='<i4')
  for b in range(B):
   start=b*T;stop=min(start+T,N);part=f[:,start:stop];idx=np.argmax(part,axis=1);v[:,b]=part[np.arange(M),idx];ix[:,b]=idx+start
  ids=np.argmax(f,axis=1).astype('<i8')
  row.update(expected_V=put(name+'.V.f32',v),expected_I=put(name+'.I.i32',ix),expected_ids=put(name+'.ids.i64',ids),cast_sha256=hashlib.sha256(f.tobytes()).hexdigest(),host_V_zero_sign_policy='either zero sign for numeric oracle; cross-arm compact V raw bits must match exactly',compact_shape=[M,B],compact_strides=[B,1])
 else:row.update(nonfinite_flat_indices=indices,nonfinite_patterns=[int(a.reshape(-1)[k])for k in indices],expected_rejected_before_blocks_and_finish=True)
 rows.append(row)
assert np.unique(adv[0]).size==65280 and [int(x)for x in np.argmax((adv.astype(np.uint32)<<16).view('<f4'),axis=1)[:6]]==[32639,0,N-1,0,N-1,1023]
manifest=dict(schema='tileir.logits.bf16.synthetic-fixtures.v1',generator=dict(file=str(R/'make_fixtures.py'),sha256=sha(R/'make_fixtures.py')),numpy_version=np.__version__,synthetic_not_saved_model_logits=True,shape=[M,N],finite_status_words=7760,greedy_blocks_per_row=B,last_greedy_tile_valid=512,last_greedy_tile_masked=512,fixtures=rows,nominal_formula='u=u32(i*1664525+1013904223);u^=u>>16;bits=(u&0x8000)|(0x3b80+((u>>16)%1408))',coverage=['all65280finiteBF16encodings in adversarial row0','alternating signed zero and lowest-index ties','all-negative rows and last-valid-half-tile winners','cross-block and final-half-tile tied maxima','signed signaling/quiet NaN,+Inf,-Inf at fixed global/row/chunk/tail boundaries'],limits=['CPU oracle/generator only; no GPU execution','Host max oracle permits either signed zero value; raw GPU V comparison across arms remains strict','One deterministic nominal finite fixture for timing; not real logits or cache-neutral evidence','Sentinels/correct values do not prove absence of out-of-bounds reads'])
with(R/'FIXTURE.json').open('x')as f:json.dump(manifest,f,indent=2);f.write('\n')
print(json.dumps(dict(complete=True,fixture_sha256=sha(R/'FIXTURE.json'),cases=len(rows),input_bytes=sum(x['input']['bytes']for x in rows),files=len(list(D.iterdir())),gpu_executed=False)))
