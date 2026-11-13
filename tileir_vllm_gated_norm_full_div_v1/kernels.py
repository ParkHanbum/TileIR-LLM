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
 variance=ct.sum(x*x,axis=0)/128.0
 normalized=x*ct.rsqrt(variance+Epsilon)
 weighted=normalized*w
 sigmoid=ct.truediv(1.0,1.0+ct.exp2((-z)*1.4426950408889634),rounding_mode=ct.RoundingMode.FULL)
 gate=z*sigmoid
 output=(weighted*gate).astype(ct.bfloat16)
 ct.scatter(Y,(token,head,col),output,check_bounds=True)
