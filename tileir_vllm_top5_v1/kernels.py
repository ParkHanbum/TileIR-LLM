# SPDX-License-Identifier: Apache-2.0
# Independent two-stage top5 selection; values descending, ties index ascending.
import cuda.tile as ct
@ct.kernel
def blocks(X,V,I):
 row=ct.bid(0);b=ct.bid(1);col=b*1024+ct.arange(1024,dtype=ct.int32)
 x=ct.gather(X,(row,col),check_bounds=True,padding_value=float('-inf')).astype(ct.float32)
 active=col<248320
 for j in range(5):
  mx=ct.max(ct.where(active,x,float('-inf')),axis=0)
  idx=ct.min(ct.where(active&(x==mx),col,2147483647),axis=0)
  ct.scatter(V,(row,b*5+j),mx,check_bounds=True);ct.scatter(I,(row,b*5+j),idx,check_bounds=True)
  active=active&(col!=idx)
@ct.kernel
def finish(V,I,Y):
 row=ct.bid(0);p=ct.arange(2048,dtype=ct.int32)
 v=ct.gather(V,(row,p),check_bounds=True,padding_value=float('-inf'))
 i=ct.gather(I,(row,p),check_bounds=True,padding_value=2147483647)
 active=p<1215
 for j in range(5):
  mx=ct.max(ct.where(active,v,float('-inf')),axis=0)
  idx=ct.min(ct.where(active&(v==mx),i,2147483647),axis=0)
  ct.scatter(Y,(row,j),idx.astype(ct.int64),check_bounds=True)
  active=active&(i!=idx)
