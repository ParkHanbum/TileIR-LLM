# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the vLLM project; original interface Copyright(c)2024 Tri Dao.
# Independent TileIR translation of the scoped public causal_conv1d_fn semantics.
import cuda.tile as ct
from cuda.tile import RoundingMode as RM
@ct.kernel
def prefill_kernel(X,S,W,I,Y):
 t=ct.bid(0);b=t//53;local=t%53;d=ct.bid(1)*256+ct.arange(256,dtype=ct.int32)
 acc=ct.full((256,),0.0,ct.float32)
 pos0=ct.where(local>=3,t-3,b*53)
 v0=ct.gather(X,(d,pos0))
 v0=ct.where(local>=3,v0,ct.astype(0.0,ct.bfloat16))
 w0=ct.gather(W,(d,0))
 p0=ct.astype(ct.astype(ct.mul(ct.astype(v0,ct.float32),ct.astype(w0,ct.float32),rounding_mode=RM.RN),ct.bfloat16),ct.float32)
 acc=ct.add(acc,p0,rounding_mode=RM.RN)
 pos1=ct.where(local>=2,t-2,b*53)
 v1=ct.gather(X,(d,pos1))
 v1=ct.where(local>=2,v1,ct.astype(0.0,ct.bfloat16))
 w1=ct.gather(W,(d,1))
 p1=ct.astype(ct.astype(ct.mul(ct.astype(v1,ct.float32),ct.astype(w1,ct.float32),rounding_mode=RM.RN),ct.bfloat16),ct.float32)
 acc=ct.add(acc,p1,rounding_mode=RM.RN)
 pos2=ct.where(local>=1,t-1,b*53)
 v2=ct.gather(X,(d,pos2))
 v2=ct.where(local>=1,v2,ct.astype(0.0,ct.bfloat16))
 w2=ct.gather(W,(d,2))
 p2=ct.astype(ct.astype(ct.mul(ct.astype(v2,ct.float32),ct.astype(w2,ct.float32),rounding_mode=RM.RN),ct.bfloat16),ct.float32)
 acc=ct.add(acc,p2,rounding_mode=RM.RN)
 pos3=ct.where(local>=0,t-0,b*53)
 v3=ct.gather(X,(d,pos3))
 v3=ct.where(local>=0,v3,ct.astype(0.0,ct.bfloat16))
 w3=ct.gather(W,(d,3))
 p3=ct.astype(ct.astype(ct.mul(ct.astype(v3,ct.float32),ct.astype(w3,ct.float32),rounding_mode=RM.RN),ct.bfloat16),ct.float32)
 acc=ct.add(acc,p3,rounding_mode=RM.RN)
 neg=ct.sub(0.0,acc,rounding_mode=RM.RN)
 exp=ct.exp2(ct.mul(neg,1.4426950408889634,rounding_mode=RM.RN))
 y=ct.truediv(acc,ct.add(exp,1.0,rounding_mode=RM.RN),rounding_mode=RM.FULL)
 ct.scatter(Y,(d,t),ct.astype(y,ct.bfloat16))
 if local==52:
  slot=ct.gather(I,b)
  ct.scatter(S,(slot,d,0),v1);ct.scatter(S,(slot,d,1),v2);ct.scatter(S,(slot,d,2),v3)
