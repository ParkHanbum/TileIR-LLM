# SPDX-License-Identifier: Apache-2.0
# Independent TileIR expression of public vLLM integer metadata operation.
import cuda.tile as ct
@ct.kernel
def kernel(Mapping,Starts,Computed,Pos,Seq,N):
 req=ct.bid(0)
 if req<N:
  idx=ct.gather(Mapping,req,check_bounds=True)
  computed=ct.gather(Computed,idx,check_bounds=True)
  start=ct.gather(Starts,req,check_bounds=True)
  end=ct.gather(Starts,req+1,check_bounds=True)
  length=end-start
  ct.scatter(Seq,req,computed+length,check_bounds=True)
  for base in range(0,length,1024):
   local=base+ct.arange(1024,dtype=ct.int32)
   index=ct.where(local<length,start+local,Pos.shape[0])
   ct.scatter(Pos,index,(computed+local).astype(ct.int64),check_bounds=True)
 else:ct.scatter(Seq,req,0,check_bounds=True)
