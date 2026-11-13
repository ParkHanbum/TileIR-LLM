# SPDX-License-Identifier: Apache-2.0
# Independent TileIR expression of public vLLM metadata/min-token algorithms.
# Copyright contributors to the vLLM project. Source provenance in REPORT.md.
import cuda.tile as ct
@ct.kernel
def rope(Out,Prefill,Delta,Idx,Starts,Lens,Computed):
 b=ct.bid(0);axis=ct.bid(1);r=ct.gather(Idx,b)
 begin=ct.gather(Starts,b);end=ct.gather(Starts,b+1)
 count=ct.gather(Computed,r);plen=ct.gather(Lens,r);delta=ct.gather(Delta,r)
 for base in range(0,end-begin,128):
  j=base+ct.arange(128,dtype=ct.int32);valid=j<end-begin;orig=count+j
  if count<plen:val=ct.gather(Prefill,(r*3+axis,orig),mask=valid,padding_value=0)
  else:val=orig+delta
  ct.scatter(Out,(axis,begin+j),val.astype(ct.int64),mask=valid)
@ct.kernel
def combine(Input,Idx,Last,Starts,Seq,Prefill,Cu,Out):
 b=ct.bid(0);r=ct.gather(Idx,b);end=ct.gather(Starts,b+1);start=ct.gather(Cu,b)
 ct.scatter(Out,start,(end-1).astype(ct.int64))
 seq=ct.gather(Seq,b);plen=ct.gather(Prefill,r)
 if (seq>plen)&(seq-1>=plen):
  tok=ct.gather(Last,(r,0));ct.scatter(Input,end-1,tok.astype(ct.int32))
@ct.kernel
def bias(Logits,Idx,Pos,Minimum,Counts,Stop):
 row=ct.bid(0);r=ct.gather(Idx,row);pos=ct.gather(Pos,row);minimum=ct.gather(Minimum,r);n=ct.gather(Counts,r)
 if (n>0)&(pos+1<minimum):
  j=ct.arange(128,dtype=ct.int32);ids=ct.gather(Stop,(r,j),mask=j<n,padding_value=0)
  ct.scatter(Logits,(row,ids),ct.full((128,),float('-inf'),ct.float32),mask=j<n)
