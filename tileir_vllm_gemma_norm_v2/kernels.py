# SPDX-License-Identifier: BSD-3-Clause
# Public ATen Reduce.cuh vector4 lane order, expressed with cuTile RN ops.
import cuda.tile as ct
from cuda.tile import RoundingMode as RM
@ct.kernel
def gemma_norm_kernel(X,W,R,Y,RO,Epsilon,HasResidual:ct.Constant[bool],BLOCK:ct.Constant[int]):
 row=ct.bid(0);lane=ct.arange(32,dtype=ct.int32)
 a0=ct.full((32,),0,ct.float32)
 a1=ct.full((32,),0,ct.float32)
 a2=ct.full((32,),0,ct.float32)
 a3=ct.full((32,),0,ct.float32)
 for iteration in range(20):
  j0=lane*4+iteration*128+0
  x0=ct.gather(X,(row,j0),check_bounds=False).astype(ct.float32)
  v0=x0
  if HasResidual:
   r0=ct.gather(R,(row,j0),check_bounds=False).astype(ct.float32)
   v0=ct.add(x0,r0,rounding_mode=RM.RN)
  sq0=ct.mul(v0,v0,rounding_mode=RM.RN)
  a0=ct.add(a0,sq0,rounding_mode=RM.RN)
  j1=lane*4+iteration*128+1
  x1=ct.gather(X,(row,j1),check_bounds=False).astype(ct.float32)
  v1=x1
  if HasResidual:
   r1=ct.gather(R,(row,j1),check_bounds=False).astype(ct.float32)
   v1=ct.add(x1,r1,rounding_mode=RM.RN)
  sq1=ct.mul(v1,v1,rounding_mode=RM.RN)
  a1=ct.add(a1,sq1,rounding_mode=RM.RN)
  j2=lane*4+iteration*128+2
  x2=ct.gather(X,(row,j2),check_bounds=False).astype(ct.float32)
  v2=x2
  if HasResidual:
   r2=ct.gather(R,(row,j2),check_bounds=False).astype(ct.float32)
   v2=ct.add(x2,r2,rounding_mode=RM.RN)
  sq2=ct.mul(v2,v2,rounding_mode=RM.RN)
  a2=ct.add(a2,sq2,rounding_mode=RM.RN)
  j3=lane*4+iteration*128+3
  x3=ct.gather(X,(row,j3),check_bounds=False).astype(ct.float32)
  v3=x3
  if HasResidual:
   r3=ct.gather(R,(row,j3),check_bounds=False).astype(ct.float32)
   v3=ct.add(x3,r3,rounding_mode=RM.RN)
  sq3=ct.mul(v3,v3,rounding_mode=RM.RN)
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
 factor=ct.full((),0.000390625,ct.float32)
 variance=ct.mul(summed,factor,rounding_mode=RM.RN)
 eps_added=ct.add(variance,Epsilon,rounding_mode=RM.RN)
 rstd=ct.rsqrt(eps_added)
 col=ct.arange(BLOCK,dtype=ct.int32);mask=col<2560
 x=ct.gather(X,(row,col),mask=mask,padding_value=0,check_bounds=True).astype(ct.float32)
 w=ct.gather(W,col,mask=mask,padding_value=0,check_bounds=True).astype(ct.float32)
 value=x
 if HasResidual:
  r=ct.gather(R,(row,col),mask=mask,padding_value=0,check_bounds=True).astype(ct.float32)
  value=ct.add(x,r,rounding_mode=RM.RN)
  ct.scatter(RO,(row,col),value.astype(ct.bfloat16),mask=mask,check_bounds=True)
 weight=ct.add(w,1.0,rounding_mode=RM.RN)
 normalized=ct.mul(value,rstd,rounding_mode=RM.RN)
 weighted=ct.mul(normalized,weight,rounding_mode=RM.RN)
 ct.scatter(Y,(row,col),weighted.astype(ct.bfloat16),mask=mask,check_bounds=True)
