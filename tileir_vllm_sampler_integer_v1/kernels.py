# SPDX-License-Identifier: Apache-2.0
import cuda.tile as ct
@ct.kernel
def gather_rows(x,indices,out,n:ct.Constant[int]):
 lane=ct.arange(32,dtype=ct.int32)
 index=ct.gather(indices,lane,check_bounds=True)
 normalized=ct.where(index<0,index+n,index)
 value=ct.gather(x,normalized,check_bounds=True)
 ct.scatter(out,lane,value,check_bounds=True)
@ct.kernel
def ones(out):
 lane=ct.arange(32,dtype=ct.int32)
 ct.scatter(out,lane,ct.full((32,),1,dtype=ct.int32),check_bounds=True)
