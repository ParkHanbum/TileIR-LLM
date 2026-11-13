# SPDX-License-Identifier: Apache-2.0
# Public semantic reference: vLLM GemmaRMSNorm and ir.ops.layernorm.
# Independent cuTile expression, not vendor binary-derived.
import cuda.tile as ct
@ct.kernel
def gemma_norm_kernel(X,W,R,Y,RO,Epsilon,HasResidual:ct.Constant[bool],BLOCK:ct.Constant[int]):
 row=ct.bid(0)
 col=ct.arange(BLOCK,dtype=ct.int32)
 active=col<2560
 value=ct.gather(X,(row,col),mask=active,padding_value=0,check_bounds=True).astype(ct.float32)
 if HasResidual:
  residual=ct.gather(R,(row,col),mask=active,padding_value=0,check_bounds=True).astype(ct.float32)
  value=value+residual
  ct.scatter(RO,(row,col),value.astype(ct.bfloat16),mask=active,check_bounds=True)
 weight=ct.gather(W,col,mask=active,padding_value=0,check_bounds=True).astype(ct.float32)+1.0
 variance=ct.sum(value*value,axis=0)/2560.0
 normalized=value*ct.rsqrt(variance+Epsilon)
 output=(normalized*weight).astype(ct.bfloat16)
 ct.scatter(Y,(row,col),output,mask=active,check_bounds=True)
