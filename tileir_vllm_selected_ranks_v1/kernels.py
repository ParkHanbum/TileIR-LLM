# SPDX-License-Identifier: Apache-2.0
import cuda.tile as ct
@ct.kernel
def partial(X,IDs,C):
 row=ct.bid(0);b=ct.bid(1);col=b*8192+ct.arange(8192,dtype=ct.int32)
 chosen=ct.gather(IDs,row,check_bounds=True);score=ct.gather(X,(row,chosen),check_bounds=True)
 values=ct.gather(X,(row,col),check_bounds=True,padding_value=float('-inf'))
 count=ct.sum(((col<248320)&(values>=score)).astype(ct.int32),axis=0)
 ct.scatter(C,(row,b),count,check_bounds=True)
@ct.kernel
def finish(C,Y):
 row=ct.bid(0);b=ct.arange(32,dtype=ct.int32)
 c=ct.gather(C,(row,b),check_bounds=True,padding_value=0)
 ct.scatter(Y,row,ct.sum(c,axis=0).astype(ct.int64),check_bounds=True)
