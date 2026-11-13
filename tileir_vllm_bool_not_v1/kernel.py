# SPDX-License-Identifier: Apache-2.0
import cuda.tile as ct
@ct.kernel
def invert_bool(mask,out):
 i=ct.arange(32,dtype=ct.int32)
 x=ct.gather(mask,i,check_bounds=True)
 ct.scatter(out,i,x==False,check_bounds=True)
