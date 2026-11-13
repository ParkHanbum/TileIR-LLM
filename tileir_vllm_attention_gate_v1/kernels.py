# SPDX-License-Identifier: Apache-2.0
# Independent expression of public vLLM attention sigmoid-gate semantics.
import cuda.tile as ct
@ct.kernel
def kernel(X,G,Y):
 row=ct.bid(0);col=ct.arange(4096,dtype=ct.int32)
 x=ct.gather(X,(row,col),check_bounds=True).astype(ct.float32)
 g=ct.gather(G,(row,col),check_bounds=True).astype(ct.float32)
 # Eager torch.sigmoid(BF16) materializes BF16 before torch.mul.
 s=(1.0/(1.0+ct.exp(-g))).astype(ct.bfloat16).astype(ct.float32)
 y=(x*s).astype(ct.bfloat16)
 ct.scatter(Y,(row,col),y,check_bounds=True)
