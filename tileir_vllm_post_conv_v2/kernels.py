# SPDX-License-Identifier: Apache-2.0
# Public vLLM fused_gdn_prefill_post_conv semantics, independently expressed in cuTile.
# Source contributor notice and provenance in README.md / LICENSE.
import cuda.tile as ct
from cuda.tile import RoundingMode as RM
@ct.kernel
def qk_kernel(X,Q,K):
 t=ct.bid(0);h=ct.bid(1);kind=ct.bid(2);lane=ct.arange(16,dtype=ct.int32)
 base=kind*2048+h*128
 x=ct.gather(X,(t,base+lane*8),check_bounds=False).astype(ct.float32)
 acc=ct.mul(x,x,rounding_mode=RM.RN)
 for i in range(1,8):
  x=ct.gather(X,(t,base+lane*8+i),check_bounds=False).astype(ct.float32)
  acc=ct.add(ct.mul(x,x,rounding_mode=RM.RN),acc,rounding_mode=RM.RN)
 acc=ct.add(ct.extract(acc,(0,),(8,)),ct.extract(acc,(1,),(8,)),rounding_mode=RM.RN)
 acc=ct.add(ct.extract(acc,(0,),(4,)),ct.extract(acc,(1,),(4,)),rounding_mode=RM.RN)
 acc=ct.add(ct.extract(acc,(0,),(2,)),ct.extract(acc,(1,),(2,)),rounding_mode=RM.RN)
 acc=ct.add(ct.extract(acc,(0,),(1,)),ct.extract(acc,(1,),(1,)),rounding_mode=RM.RN)
 summed=ct.reshape(acc,())
 denom=ct.sqrt(ct.add(summed,1e-6,rounding_mode=RM.RN),rounding_mode=RM.APPROX,flush_to_zero=True)
 inv=ct.truediv(1.0,denom,rounding_mode=RM.FULL)
 col=ct.arange(128,dtype=ct.int32)
 x=ct.gather(X,(t,base+col),check_bounds=False).astype(ct.float32)
 out=ct.mul(x,inv,rounding_mode=RM.RN).astype(ct.bfloat16)
 if kind==0:ct.scatter(Q,(t,h,col),out,check_bounds=False)
 else:ct.scatter(K,(t,h,col),out,check_bounds=False)
@ct.kernel
def vg_kernel(X,A,B,AL,DT,V,G,Beta):
 t=ct.bid(0);h=ct.bid(1);col=ct.arange(128,dtype=ct.int32)
 value=ct.gather(X,(t,4096+h*128+col),check_bounds=False)
 ct.scatter(V,(t,h,col),value,check_bounds=False)
 a=ct.gather(A,(t,h),check_bounds=False).astype(ct.float32)
 b=ct.gather(B,(t,h),check_bounds=False).astype(ct.float32)
 al=ct.gather(AL,h,check_bounds=False).astype(ct.float32)
 dt=ct.gather(DT,h,check_bounds=False).astype(ct.float32)
 x=ct.add(a,dt,rounding_mode=RM.RN)
 neg_exp=ct.exp2(ct.mul(-x,1.4426950408889634,rounding_mode=RM.RN))
 pos_exp=ct.exp2(ct.mul(x,1.4426950408889634,rounding_mode=RM.RN))
 sp_pos=ct.add(x,ct.log(ct.add(1.0,neg_exp,rounding_mode=RM.RN)),rounding_mode=RM.RN)
 sp_neg=ct.log(ct.add(1.0,pos_exp,rounding_mode=RM.RN))
 sp=ct.where(x>0,sp_pos,sp_neg);sp=ct.where(x<=20.0,sp,x)
 g=ct.mul(-ct.exp2(ct.mul(al,1.4426950408889634,rounding_mode=RM.RN)),sp,rounding_mode=RM.RN)
 beta=ct.truediv(1.0,ct.add(1.0,ct.exp2(ct.mul(-b,1.4426950408889634,rounding_mode=RM.RN)),rounding_mode=RM.RN),rounding_mode=RM.FULL)
 ct.scatter(G,(t,h),g,check_bounds=False);ct.scatter(Beta,(t,h),beta,check_bounds=False)
