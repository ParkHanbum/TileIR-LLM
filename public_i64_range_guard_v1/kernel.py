# SPDX-License-Identifier: Apache-2.0
# Own implementation; documented public cuTile gather/full/compare/sum/scatter API.
import cuda.tile as ct
@ct.kernel
def range_status(x,status,n:ct.Constant[int],width:ct.Constant[int],lower:ct.Constant[int],upper:ct.Constant[int]):
 i=ct.arange(width,dtype=ct.int32)
 value=ct.gather(x,i,check_bounds=True)
 lo=ct.full((width,),lower,dtype=ct.int64)
 hi=ct.full((width,),upper,dtype=ct.int64)
 valid=(value>=lo)&(value<hi)
 bad=ct.where(i<n,ct.where(valid,0,1),0).astype(ct.int32)
 ct.scatter(status,0,ct.sum(bad,axis=0),check_bounds=True)
