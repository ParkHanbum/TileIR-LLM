# SPDX-License-Identifier: Apache-2.0
# Public vLLM/FLA chunk_o.py adaptation; original MIT in LICENSE-FLA-MIT.txt.
import cuda.tile as ct
from cuda.tile import RoundingMode as RM
@ct.kernel
def output(Q,K,V,H,G,O):
 seq=ct.bid(0);head=ct.bid(1);part=ct.bid(2)
 t=ct.arange(64,dtype=ct.int32);d=ct.arange(32,dtype=ct.int32);vv=part*32+d
 zO=ct.full((64,32),0,dtype=ct.float32);zA=ct.full((64,64),0,dtype=ct.float32)
 bo=zO;ba=zA
 for block in range(4):
  kk=block*32+d
  q=ct.gather(Q,(0,seq*53+t[:,None],head//2,kk[None,:]),mask=t[:,None]<53,padding_value=0,check_bounds=False)
  k=ct.gather(K,(0,seq*53+t[None,:],head//2,kk[:,None]),mask=t[None,:]<53,padding_value=0,check_bounds=False)
  h=ct.gather(H,(0,seq,head,vv[:,None],kk[None,:]),check_bounds=False)
  bo=ct.mma(q,ct.transpose(h),bo)
  ba=ct.mma(q,k,ba)
 g=ct.gather(G,(0,seq*53+t,head),mask=t<53,padding_value=0,check_bounds=False)
 eg=ct.exp2(ct.mul(g,1.4426950408889634,rounding_mode=RM.RN))
 diff=ct.sub(g[:,None],g[None,:],rounding_mode=RM.RN)
 decay=ct.exp2(ct.mul(diff,1.4426950408889634,rounding_mode=RM.RN))
 bo=ct.mul(bo,eg[:,None],rounding_mode=RM.RN);ba=ct.mul(ba,decay,rounding_mode=RM.RN)
 mask=(t[:,None]>=t[None,:])&(t[:,None]<53)&(t[None,:]<53)
 ba=ct.where(mask,ba,0.0).astype(ct.bfloat16,rounding_mode=RM.RN)
 v=ct.gather(V,(0,seq*53+t[:,None],head,vv[None,:]),mask=t[:,None]<53,padding_value=0,check_bounds=False)
 product=ct.mma(ba,v,zO)
 result=bo*0.08838834764831845+product*0.08838834764831845
 ct.scatter(O,(0,seq*53+t[:,None],head,vv[None,:]),result.astype(ct.bfloat16,rounding_mode=RM.RN),mask=t[:,None]<53,check_bounds=False)
