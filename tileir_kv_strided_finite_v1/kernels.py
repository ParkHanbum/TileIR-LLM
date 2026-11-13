# SPDX-License-Identifier: Apache-2.0
import cuda.tile as ct
@ct.kernel
def count_rows(x,counts):
 row=ct.bid(0);i=ct.arange(1024,dtype=ct.int32)
 # Original tensor's offset/outer stride are carried in the public descriptor.
 v=ct.gather(x,(row,i//256,i%256),check_bounds=True).astype(ct.float32)
 finite=(v==v)&(v<float('inf'))&(v>float('-inf'))
 total=ct.sum(ct.where(finite,0,1).astype(ct.int32),axis=0)
 ct.scatter(counts,row,total,check_bounds=True)
