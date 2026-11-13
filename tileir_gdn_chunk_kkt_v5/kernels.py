# SPDX-License-Identifier: Apache-2.0
# Independent cuTile expression of public vLLM/FLA KKT. See README/licenses.
import cuda.tile as ct
from cuda.tile import RoundingMode as RM
@ct.kernel
def kkt(K,G,Beta,Cu,Indices,A,BK:ct.Constant[int]):
 chunk=ct.bid(0);head=ct.bid(1);r=ct.arange(64,dtype=ct.int32)
 seq=ct.gather(Indices,(chunk,0),check_bounds=False).astype(ct.int32)
 local=ct.gather(Indices,(chunk,1),check_bounds=False).astype(ct.int32)
 begin=ct.gather(Cu,seq,check_bounds=False);end=ct.gather(Cu,seq+1,check_bounds=False)
 t=begin+local*64+r;valid=t<end
 beta=ct.gather(Beta,(0,t,head),mask=valid,padding_value=0,check_bounds=False)
 accum=ct.full((64,64),0,dtype=ct.float32)
 for block in range(16):
  col=(block//2)*16+(block%2)+2*ct.arange(8,dtype=ct.int32)
  key=ct.gather(K,(0,t[:,None],head//2,col[None,:]),mask=valid[:,None],padding_value=0,check_bounds=False).astype(ct.float32)
  scaled=ct.mul(key,beta[:,None],rounding_mode=RM.RN,flush_to_zero=False)
  # Supported conversion: RZ truncation hypothesis; native equality remains to be tested.
  lhs=scaled.astype(ct.tfloat32,rounding_mode=RM.RZ)
  rhs=ct.transpose(key.astype(ct.tfloat32,rounding_mode=RM.RZ))
  accum=ct.mma(lhs,rhs,accum)
 gate=ct.gather(G,(0,t,head),mask=valid,padding_value=0,check_bounds=False)
 diff=ct.sub(gate[:,None],gate[None,:],rounding_mode=RM.RN,flush_to_zero=False)
 factor=ct.exp2(ct.mul(diff,1.4426950408889634,rounding_mode=RM.RN,flush_to_zero=False),flush_to_zero=False)
 value=ct.mul(accum,factor,rounding_mode=RM.RN,flush_to_zero=False)
 lower=(r[:,None]>r[None,:])&valid[:,None]&valid[None,:]
 value=ct.where(lower,value,0.0)
 ct.scatter(A,(0,t[:,None],head,r[None,:]),value,mask=valid[:,None],check_bounds=False)
