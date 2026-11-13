# SPDX-License-Identifier: Apache-2.0
# Independent public greedy maximum/index reduction, no random sampling.
import cuda.tile as ct
@ct.kernel
def blocks(X,V,I):
 row=ct.bid(0);b=ct.bid(1);col=b*1024+ct.arange(1024,dtype=ct.int32)
 x=ct.gather(X,(row,col),check_bounds=True,padding_value=float('-inf'))
 mx=ct.max(x,axis=0)
 idx=ct.min(ct.where((col<248320)&(x==mx),col,2147483647),axis=0)
 ct.scatter(V,(row,b),mx,check_bounds=True);ct.scatter(I,(row,b),idx,check_bounds=True)
@ct.kernel
def finish(V,I,Y):
 row=ct.bid(0);b=ct.arange(256,dtype=ct.int32)
 v=ct.gather(V,(row,b),check_bounds=True,padding_value=float('-inf'))
 i=ct.gather(I,(row,b),check_bounds=True,padding_value=2147483647)
 mx=ct.max(v,axis=0);idx=ct.min(ct.where(v==mx,i,2147483647),axis=0)
 ct.scatter(Y,row,idx.astype(ct.int64),check_bounds=True)
