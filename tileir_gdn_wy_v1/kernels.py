# SPDX-License-Identifier: Apache-2.0
# Adapted from public vLLM/FLA wy_fast.py; see LICENSE-FLA-MIT.txt.
import cuda.tile as ct
from cuda.tile import RoundingMode as RM
@ct.kernel
def wy(K,V,B,G,A,W,U):
 seq=ct.bid(0);h=ct.bid(1);part=ct.bid(2)
 t=ct.arange(64,dtype=ct.int32);c=part*64+ct.arange(64,dtype=ct.int32)
 beta=ct.gather(B,(0,seq*53+t,h),mask=t<53,padding_value=0,check_bounds=False)
 g=ct.gather(G,(0,seq*53+t,h),mask=t<53,padding_value=0,check_bounds=False)
 eg=ct.exp2(ct.mul(g,1.4426950408889634,rounding_mode=RM.RN))
 a=ct.gather(A,(0,seq*53+t[:,None],h,t[None,:]),mask=t[:,None]<53,padding_value=0,check_bounds=False)
 v=ct.gather(V,(0,seq*53+t[:,None],h,c[None,:]),mask=t[:,None]<53,padding_value=0,check_bounds=False)
 k=ct.gather(K,(0,seq*53+t[:,None],h//2,c[None,:]),mask=t[:,None]<53,padding_value=0,check_bounds=False)
 vb=ct.mul(v.astype(ct.float32),beta[:,None],rounding_mode=RM.RN).astype(ct.bfloat16,rounding_mode=RM.RN)
 kb=ct.mul(ct.mul(k.astype(ct.float32),beta[:,None],rounding_mode=RM.RN),eg[:,None],rounding_mode=RM.RN).astype(ct.bfloat16,rounding_mode=RM.RN)
 zero=ct.full((64,64),0,dtype=ct.float32)
 u=ct.mma(a,vb,zero);w=ct.mma(a,kb,zero)
 ct.scatter(U,(0,seq*53+t[:,None],h,c[None,:]),u.astype(ct.bfloat16,rounding_mode=RM.RN),mask=t[:,None]<53,check_bounds=False)
 ct.scatter(W,(0,seq*53+t[:,None],h,c[None,:]),w.astype(ct.bfloat16,rounding_mode=RM.RN),mask=t[:,None]<53,check_bounds=False)
