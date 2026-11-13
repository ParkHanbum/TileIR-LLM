# SPDX-License-Identifier: Apache-2.0
# Public vLLM buffer_utils._apply_write_kernel logical adaptation.
# Copyright contributors to the vLLM project.
import cuda.tile as ct
@ct.kernel
def write_segments(out,indices,starts,contents,ends,STRIDE:ct.Constant[int]):
 p=ct.bid(0);row=ct.gather(indices,p).astype(ct.int64);start=ct.gather(starts,p)
 begin=0
 if p>0:begin=ct.gather(ends,p-1)
 end=ct.gather(ends,p)
 for base in range(0,end-begin,1024):
  j=base+ct.arange(1024,dtype=ct.int32);mask=j<end-begin
  v=ct.gather(contents,begin+j,mask=mask,padding_value=0)
  ct.scatter(out,row*STRIDE+start+j,v,mask=mask)
