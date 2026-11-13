# SPDX-License-Identifier: Apache-2.0
import cuda.tile as ct
@ct.kernel
def construct(O,arange:ct.Constant[bool]):
 i=ct.arange(64,dtype=ct.int32)
 value=i if arange else ct.full((64,),0,ct.int32)
 ct.scatter(O,i,value,check_bounds=True)
