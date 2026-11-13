"""Public algorithm CPU byte oracle. Never uses native GPU as an oracle."""
import struct

def read(raw,code):return list(struct.unpack('<'+code*(len(raw)//struct.calcsize(code)),raw))
def put(raw,index,value,code):struct.pack_into('<'+code,raw,index*struct.calcsize(code),value)
def expected(label,raw,a):
 if label=='prepare_rope_positions':
  out=bytearray(raw['self.positions']);idx=read(raw['idx_mapping'],'q');q=read(raw['query_start_loc'],'i');lens=read(raw['prefill_lens'],'i');computed=read(raw['num_computed_tokens'],'i');delta=read(raw['self.prefill_delta.gpu'],'i');p=read(raw['self.prefill_positions.gpu'],'i')
  for b,r in enumerate(idx):
   for j in range(q[b+1]-q[b]):
    for axis in range(3):put(out,axis*2049+q[b]+j,p[(r*3+axis)*128+computed[r]+j] if computed[r]<lens[r] else computed[r]+j+delta[r],'q')
  return {'self.positions':out},None
 if label=='combine_sampled_and_draft_tokens':
  out=bytearray(raw['input_ids']);idx=read(raw['idx_mapping'],'q');q=read(raw['query_start_loc'],'i');seq=read(raw['seq_lens'],'i');prefill=read(raw['prefill_len'],'i');last=read(raw['last_sampled_tokens'],'q');ret=[]
  for b,r in enumerate(idx):
   ret.append(q[b+1]-1)
   if seq[b]>prefill[r] and seq[b]-1>=prefill[r]:put(out,q[b+1]-1,last[r],'i')
  return {'input_ids':out},struct.pack('<'+'q'*32,*ret)
 if label=='bias':
  out=bytearray(raw['logits']);idx=read(raw['expanded_idx_mapping'],'q');pos=read(raw['pos'],'q');minimum=read(raw['min_lens'],'i');count=read(raw['num_stop_token_ids'],'i');ids=read(raw['stop_token_ids'],'i')
  for b,r in enumerate(idx):
   if pos[b]+1<minimum[r]:
    for j in range(count[r]):put(out,b*248320+ids[r*128+j],float('-inf'),'f')
  return {'logits':out},None
 if label=='post_update':
  names=('num_computed_tokens','last_sampled_tokens','output_bin_counts','all_token_ids','total_len');out={n:bytearray(raw[n])for n in names}
  idx=read(raw['idx_mapping'],'q');ns=read(raw['num_sampled'],'i');nr=read(raw['num_rejected'],'i');q=read(raw['query_start_loc'],'i');sampled=read(raw['sampled_tokens'],'q');total=read(raw['total_len'],'i');computed=read(raw['num_computed_tokens'],'i')
  for b,r in enumerate(idx):
   if r<0:continue
   if ns[b]:
    token=sampled[b];put(out['last_sampled_tokens'],r,token,'q');put(out['total_len'],r,total[r]+1,'i');put(out['all_token_ids'],r*128+total[r],token,'i');index=r*248320+token;old=struct.unpack_from('<i',out['output_bin_counts'],index*4)[0];put(out['output_bin_counts'],index,old+1,'i')
   delta=q[b+1]-q[b]-nr[b]
   if delta:put(out['num_computed_tokens'],r,computed[r]+delta,'i')
  return out,None
 raise ValueError('Unknown helper')
