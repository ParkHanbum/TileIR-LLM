# SPDX-License-Identifier: Apache-2.0
import cuda.tile as ct
@ct.kernel
def scatter_accepted(indices,sampled,accepted):
 row=ct.arange(32,dtype=ct.int32)
 idx=ct.gather(indices,row,check_bounds=True)
 n=ct.gather(sampled,row,check_bounds=True)
 value=ct.maximum(n,1)
 ct.scatter(accepted,idx,value,check_bounds=True)
