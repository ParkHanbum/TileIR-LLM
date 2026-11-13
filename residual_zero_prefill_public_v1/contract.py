# SPDX-License-Identifier: Apache-2.0
# Public vLLM semantics modeled independently for CPU fixtures.
def prefill_reference(input_ids,next_tokens,mapping,query_starts,all_ids,lengths,computed):
 assert len(query_starts)==len(mapping)+1
 for b,r in enumerate(mapping):
  assert 0<=r<len(all_ids)
  n=computed[r];limit=lengths[r]
  if n>=limit:continue
  start,end=query_starts[b:b+2];assert 0<=start<=end<=len(input_ids)
  assert 0<=n and n+end-start<=len(all_ids[r])
  input_ids[start:end]=all_ids[r][n:n+end-start]
  for lookahead,output in enumerate(next_tokens):
   pos=n+end-start+lookahead;output[r]=all_ids[r][pos] if pos<limit else 0

def normalize_segments(descriptors):
 # Each descriptor binds a public owned allocation, base/start/stride/page in bytes.
 merged={}
 for d in descriptors:
  owner,base,size,start,stride,page,blocks=d
  assert size>0 and blocks>0 and page>0 and stride>=page
  assert (base+start)%4==0 and stride%4==page%4==0 and start>=0
  assert start+(blocks-1)*stride+page<=size
  key=(owner,base+start)
  if key in merged:
   old=merged[key];assert old[4]==stride and old[6]==blocks
   page=max(old[5],page)
  merged[key]=(owner,base,size,start,stride,page,blocks)
 return list(merged.values())

def zero_reference(owners,segments,ids):
 # Byte-level union of selected pages; padding and other allocations unchanged.
 for owner,base,size,start,stride,page,blocks in normalize_segments(segments):
  assert len(owners[owner])==size and all(type(i)is int and 0<=i<blocks for i in ids)
  for i in ids:owners[owner][start+i*stride:start+i*stride+page]=bytes(page)
