# SPDX-License-Identifier: Apache-2.0
# Independent cuTile expression from public vLLM RMSNormGated mathematical contract.
import cuda.tile as ct
@ct.kernel
def gated_norm_kernel(X,Z,W,Y,Epsilon):
 row=ct.bid(0);token=row//32;head=row%32
 col=ct.arange(128,dtype=ct.int32)
 x=ct.gather(X,(token,head,col),check_bounds=True).astype(ct.float32)
 z=ct.gather(Z,(token,head,col),check_bounds=True).astype(ct.float32)
 w=ct.gather(W,col,check_bounds=True).astype(ct.float32)
 lane=ct.arange(16,dtype=ct.int32)
 v0=ct.gather(X,(token,head,lane*8+0),check_bounds=True).astype(ct.float32)
 s0=ct.mul(v0,v0,rounding_mode=ct.RoundingMode.RN)
 v1=ct.gather(X,(token,head,lane*8+1),check_bounds=True).astype(ct.float32)
 s1=ct.mul(v1,v1,rounding_mode=ct.RoundingMode.RN)
 v2=ct.gather(X,(token,head,lane*8+2),check_bounds=True).astype(ct.float32)
 s2=ct.mul(v2,v2,rounding_mode=ct.RoundingMode.RN)
 v3=ct.gather(X,(token,head,lane*8+3),check_bounds=True).astype(ct.float32)
 s3=ct.mul(v3,v3,rounding_mode=ct.RoundingMode.RN)
 v4=ct.gather(X,(token,head,lane*8+4),check_bounds=True).astype(ct.float32)
 s4=ct.mul(v4,v4,rounding_mode=ct.RoundingMode.RN)
 v5=ct.gather(X,(token,head,lane*8+5),check_bounds=True).astype(ct.float32)
 s5=ct.mul(v5,v5,rounding_mode=ct.RoundingMode.RN)
 v6=ct.gather(X,(token,head,lane*8+6),check_bounds=True).astype(ct.float32)
 s6=ct.mul(v6,v6,rounding_mode=ct.RoundingMode.RN)
 v7=ct.gather(X,(token,head,lane*8+7),check_bounds=True).astype(ct.float32)
 s7=ct.mul(v7,v7,rounding_mode=ct.RoundingMode.RN)
 total=ct.add(s1,s0,rounding_mode=ct.RoundingMode.RN)
 total=ct.add(total,s2,rounding_mode=ct.RoundingMode.RN)
 total=ct.add(total,s3,rounding_mode=ct.RoundingMode.RN)
 total=ct.add(total,s4,rounding_mode=ct.RoundingMode.RN)
 total=ct.add(total,s5,rounding_mode=ct.RoundingMode.RN)
 total=ct.add(total,s6,rounding_mode=ct.RoundingMode.RN)
 total=ct.add(total,s7,rounding_mode=ct.RoundingMode.RN)
 total=ct.add(ct.extract(total,(0,),(8,)),ct.extract(total,(1,),(8,)),rounding_mode=ct.RoundingMode.RN)
 total=ct.add(ct.extract(total,(0,),(4,)),ct.extract(total,(1,),(4,)),rounding_mode=ct.RoundingMode.RN)
 total=ct.add(ct.extract(total,(0,),(2,)),ct.extract(total,(1,),(2,)),rounding_mode=ct.RoundingMode.RN)
 total=ct.add(ct.extract(total,(0,),(1,)),ct.extract(total,(1,),(1,)),rounding_mode=ct.RoundingMode.RN)
 variance=ct.reshape(total,())/128.0
 normalized=x*ct.rsqrt(variance+Epsilon)
 weighted=normalized*w
 sigmoid=ct.truediv(1.0,1.0+ct.exp2((-z)*1.4426950408889634),rounding_mode=ct.RoundingMode.FULL)
 gate=z*sigmoid
 output=(weighted*gate).astype(ct.bfloat16)
 ct.scatter(Y,(token,head,col),output,check_bounds=True)
