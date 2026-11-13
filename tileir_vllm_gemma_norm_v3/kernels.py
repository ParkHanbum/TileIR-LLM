# SPDX-License-Identifier: BSD-3-Clause
# Public ATen order with actual FP32 storage boundaries, no debug-only outputs.
import cuda.tile as ct
from cuda.tile import RoundingMode as RM
@ct.kernel
def square_kernel(X,R,S,RO,HasResidual:ct.Constant[bool],BLOCK:ct.Constant[int]):
 row=ct.bid(0);col=ct.arange(BLOCK,dtype=ct.int32);mask=col<2560
 value=ct.gather(X,(row,col),mask=mask,padding_value=0,check_bounds=True).astype(ct.float32)
 if HasResidual:
  res=ct.gather(R,(row,col),mask=mask,padding_value=0,check_bounds=True).astype(ct.float32)
  value=ct.add(value,res,rounding_mode=RM.RN)
  ct.scatter(RO,(row,col),value.astype(ct.bfloat16),mask=mask,check_bounds=True)
 square=ct.mul(value,value,rounding_mode=RM.RN)
 ct.scatter(S,(row,col),square,mask=mask,check_bounds=True)
@ct.kernel
def mean_kernel(S,M):
 row=ct.bid(0);lane=ct.arange(32,dtype=ct.int32)
 a0=ct.full((32,),0,ct.float32)
 a1=ct.full((32,),0,ct.float32)
 a2=ct.full((32,),0,ct.float32)
 a3=ct.full((32,),0,ct.float32)
 for iteration in range(20):
  sq0=ct.gather(S,(row,lane*4+iteration*128+0),check_bounds=False)
  a0=ct.add(a0,sq0,rounding_mode=RM.RN)
  sq1=ct.gather(S,(row,lane*4+iteration*128+1),check_bounds=False)
  a1=ct.add(a1,sq1,rounding_mode=RM.RN)
  sq2=ct.gather(S,(row,lane*4+iteration*128+2),check_bounds=False)
  a2=ct.add(a2,sq2,rounding_mode=RM.RN)
  sq3=ct.gather(S,(row,lane*4+iteration*128+3),check_bounds=False)
  a3=ct.add(a3,sq3,rounding_mode=RM.RN)
 total=ct.add(a0,a1,rounding_mode=RM.RN)
 total=ct.add(total,a2,rounding_mode=RM.RN)
 total=ct.add(total,a3,rounding_mode=RM.RN)
 total=ct.add(ct.extract(total,(0,),(16,)),ct.extract(total,(1,),(16,)),rounding_mode=RM.RN)
 total=ct.add(ct.extract(total,(0,),(8,)),ct.extract(total,(1,),(8,)),rounding_mode=RM.RN)
 total=ct.add(ct.extract(total,(0,),(4,)),ct.extract(total,(1,),(4,)),rounding_mode=RM.RN)
 total=ct.add(ct.extract(total,(0,),(2,)),ct.extract(total,(1,),(2,)),rounding_mode=RM.RN)
 total=ct.add(ct.extract(total,(0,),(1,)),ct.extract(total,(1,),(1,)),rounding_mode=RM.RN)
 summed=ct.reshape(total,())
 mean=ct.mul(summed,ct.full((),0.000390625,ct.float32),rounding_mode=RM.RN)
 ct.scatter(M,row,mean,check_bounds=True)
@ct.kernel
def output_kernel(X,W,R,M,Y,Epsilon,HasResidual:ct.Constant[bool],BLOCK:ct.Constant[int]):
 row=ct.bid(0);col=ct.arange(BLOCK,dtype=ct.int32);mask=col<2560
 mean=ct.gather(M,row,check_bounds=False)
 eps=ct.add(mean,Epsilon,rounding_mode=RM.RN)
 rstd=ct.rsqrt(eps)
 value=ct.gather(X,(row,col),mask=mask,padding_value=0,check_bounds=True).astype(ct.float32)
 if HasResidual:
  res=ct.gather(R,(row,col),mask=mask,padding_value=0,check_bounds=True).astype(ct.float32)
  value=ct.add(value,res,rounding_mode=RM.RN)
 w=ct.gather(W,col,mask=mask,padding_value=0,check_bounds=True).astype(ct.float32)
 weight=ct.add(w,1.0,rounding_mode=RM.RN)
 normalized=ct.mul(value,rstd,rounding_mode=RM.RN)
 weighted=ct.mul(normalized,weight,rounding_mode=RM.RN)
 ct.scatter(Y,(row,col),weighted.astype(ct.bfloat16),mask=mask,check_bounds=True)
