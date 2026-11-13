"""Pure CPU value-domain gates. Checks are required, not inferred from metadata."""
def vector(x,n):
 if len(x)!=n or any(type(v) is not int for v in x):raise ValueError('integer vector contract')
def mapped(idx):
 vector(idx,32)
 if any(x<0 or x>=32 for x in idx):raise ValueError('state index range')
def queries(q):
 vector(q,33)
 if q[0]<0 or q[-1]>2048 or any(a>b for a,b in zip(q,q[1:])):raise ValueError('query intervals')
def rope(idx,q,lens,computed,delta):
 mapped(idx);queries(q)
 for x in (lens,computed,delta):vector(x,32)
 for b,r in enumerate(idx):
  n=q[b+1]-q[b];c=computed[r]
  if c<0 or c+n>2147483647:raise ValueError('computed arithmetic overflow')
  if c<lens[r]:
   if c+n>128:raise ValueError('prefill extent')
  elif n and not (-2147483648<=c+delta[r]<=c+n-1+delta[r]<=2147483647):raise ValueError('decode arithmetic overflow')
def combine(idx,q,seq,prefill,cu,last,num_logits,new_sampled):
 mapped(idx);queries(q)
 for x in (seq,prefill,last):vector(x,32)
 vector(cu,33)
 if type(num_logits) is not int or num_logits!=32 or type(new_sampled) is not int or new_sampled!=1:raise ValueError('only observed no-draft/one-new-token scope')
 if cu!=list(range(33)):raise ValueError('one logits row per request required')
 for b,r in enumerate(idx):
  if q[b+1]<=q[b]:raise ValueError('logits beyond query interval')
  if not (0<=seq[b]<=2147483647 and 0<=prefill[r]<=2147483647):raise ValueError('sequence domain')
  if seq[b]>prefill[r] and seq[b]-1>=prefill[r] and not -2147483648<=last[r]<=2147483647:raise ValueError('token cast range')
def bias(idx,pos,allowed,bias_counts,minimum,stop_counts,stop_ids,restore,check):
 mapped(idx)
 for x in (pos,allowed,bias_counts,minimum,stop_counts,restore):vector(x,32)
 if type(check) is not bool or check:raise ValueError('restore branch unsupported')
 if any(allowed) or any(bias_counts) or any(restore):raise ValueError('only min-token mask supported')
 if len(stop_ids)!=32 or any(len(row)!=128 for row in stop_ids):raise ValueError('stop table shape')
 for row,r in enumerate(idx):
  if not 0<=stop_counts[r]<=128 or not -9223372036854775808<=pos[row]<9223372036854775807:raise ValueError('stop count/position')
  # Native reads IDs only on active branch. Unused/uninitialized slots ignored.
  if pos[row]+1<minimum[r]:
   ids=stop_ids[r][:stop_counts[r]]
   if any(type(x) is not int or not 0<=x<248320 for x in ids):raise ValueError('stop token range')
   if len(set(ids))!=len(ids):raise ValueError('duplicate active stop IDs unsupported')
