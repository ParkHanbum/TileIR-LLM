from pathlib import Path
import torch,numpy as np,ctypes,json,hashlib
P=Path('/home/n/work/TileIR/public-tileir-blas-20261111');R=P/'gdn_solve_recurrence_cpu_v1';R.mkdir(exist_ok=False);B=P/'gdn_solve_fp32_diagnostic_v2/gdnSolveFP32Diag02';m=json.loads((P/'gdn_solve_fp32_diagnostic_v2/MANIFEST.json').read_text())
fma=ctypes.CDLL('libm.so.6').fmaf;fma.argtypes=[ctypes.c_float]*3;fma.restype=ctypes.c_float
rn=lambda x:np.float32(x)
results={}
for name,bank in m['banks'].items():
 A=torch.load(bank['A'],weights_only=True)[0,:53].numpy();native=torch.load(B/(name+'-native_probe.pt'),weights_only=True).numpy();own=torch.load(B/(name+'-own_probe.pt'),weights_only=True).numpy();res=[]
 for block in range(4):
  pred=np.zeros((32,16,16),dtype=np.float32)
  for h in range(32):
   d=np.zeros((16,16),dtype=np.float32);start=block*16;n=min(16,53-start)
   for i in range(n):
    for j in range(i):d[i,j]=rn(0)-A[start+i,h,start+j]
   for i in range(2,n):
    a=np.zeros(16,dtype=np.float32);a[:]=rn(0)-A[start+i,h,start:start+16]
    for col in range(16):
     groups=[]
     for base in (0,4,8,12):
      x=fma(float(a[base]),float(d[base,col]),float(rn(a[base+2]*d[base+2,col])))
      y=fma(float(a[base+1]),float(d[base+1,col]),float(rn(a[base+3]*d[base+3,col])))
      groups.append(rn(rn(x)+rn(y)))
     total=rn(rn(groups[0]+groups[2])+rn(groups[1]+groups[3]));predval=rn(a[col]+total)
     pred[h,i,col]=predval
    d[i]=pred[h,i]
   d=rn(d+np.eye(16,dtype=np.float32));pred[h]=d
  raw=pred.view(np.uint32)!=native[block].view(np.uint32);other=pred.view(np.uint32)!=own[block].view(np.uint32)
  res.append(dict(block=block,native_rawdiff=int(raw.sum()),own_rawdiff=int(other.sum()),maxabs=float(np.max(np.abs(pred-native[block])))))
 results[name]=res
(R/'RESULT.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
