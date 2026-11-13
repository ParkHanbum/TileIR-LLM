# SPDX-License-Identifier: Apache-2.0
import cuda.tile as ct
@ct.kernel
def count_nonfinite(x,counts,n:ct.Constant[int]):
 i=ct.bid(0)*1024+ct.arange(1024,dtype=ct.int32)
 v=ct.gather(x,i,check_bounds=True).astype(ct.float32)
 # Comparisons, not multiplication/abs/FP accumulation; ±0/subnormals are finite.
 finite=(v==v)&(v<float('inf'))&(v>float('-inf'))
 bad=ct.where(i<n,ct.where(finite,0,1),0).astype(ct.int32)
 total=ct.sum(bad,axis=0)
 ct.scatter(counts,ct.bid(0),total,check_bounds=True)
@ct.kernel
def reduce_counts(counts,status,width:ct.Constant[int]):
 i=ct.arange(width,dtype=ct.int32)
 v=ct.gather(counts,i,check_bounds=True)
 total=ct.sum(v,axis=0)
 ct.scatter(status,0,total,check_bounds=True)
