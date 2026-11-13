# SPDX-License-Identifier: Apache-2.0
# Public algorithm: vLLM _get_num_sampled_and_rejected_kernel.
# Copyright contributors to the vLLM project. Independent TileIR expression.
import cuda.tile as ct
@ct.kernel
def sample_counts(sampled,seq_lens,cu_logits,idx,prefill,rejected,BLOCK:ct.Constant[int]):
 row=ct.arange(BLOCK,dtype=ct.int32);valid=row<sampled.shape[0]
 req=ct.gather(idx,row,mask=valid,padding_value=0)
 seq=ct.gather(seq_lens,row,mask=valid,padding_value=0)
 plen=ct.gather(prefill,req,mask=valid,padding_value=0)
 count=ct.gather(sampled,row,mask=valid,padding_value=0)
 a=ct.gather(cu_logits,row,mask=valid,padding_value=0);b=ct.gather(cu_logits,row+1,mask=valid,padding_value=0)
 chunked=seq<plen
 count=ct.where(chunked,0,count);rej=ct.where(chunked,0,b-a-count)
 ct.scatter(sampled,row,count,mask=valid);ct.scatter(rejected,row,rej,mask=valid)
