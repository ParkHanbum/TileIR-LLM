# SPDX-License-Identifier: Apache-2.0
import cuda.tile as ct
@ct.kernel
def concat(C,T,O):
 i=ct.arange(256,dtype=ct.int32);row=i//6;col=i%6
 a=ct.gather(C,(row,0),check_bounds=True,padding_value=0)
 b=ct.gather(T,(row,ct.maximum(col-1,0)),check_bounds=True,padding_value=0)
 ct.scatter(O,(row,col),ct.where(col==0,a,b),check_bounds=True)
