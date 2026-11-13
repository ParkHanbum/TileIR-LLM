# SPDX-License-Identifier: Apache-2.0
# Draft public-source translation. AST checked only, no CUDA import/compile performed.
import cuda.tile as ct
@ct.kernel
def prepare_prefill_tokens(Input,Next,Mapping,Starts,All,Lengths,Computed,L:ct.Constant[int]):
 b=ct.bid(0);r=ct.gather(Mapping,(b,));limit=ct.gather(Lengths,(r,));n=ct.gather(Computed,(r,))
 if n<limit:
  start=ct.gather(Starts,(b,));end=ct.gather(Starts,(b+1,));count=end-start
  for lo in range(0,count,1024):
   x=lo+ct.arange(1024,dtype=ct.int32)
   value=ct.gather(All,(r,n+x),mask=x<count,padding_value=0,check_bounds=False)
   ct.scatter(Input,(start+x,),value,mask=x<count,check_bounds=False)
  x=ct.arange(L,dtype=ct.int32);pos=n+count+x;active=x<Next.shape[0]
  value=ct.gather(All,(r,pos),mask=active&(pos<limit),padding_value=0,check_bounds=False)
  ct.scatter(Next,(x,r),value,mask=active,check_bounds=False)
