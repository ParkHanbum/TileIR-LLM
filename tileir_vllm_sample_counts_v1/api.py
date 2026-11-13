"""Public get_num_sampled_and_rejected companion; narrow no-spec integer contract."""
def validate(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len,*,require_cuda=True):
 import torch
 vals=(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len)
 for t,dt in zip(vals,(torch.int32,torch.int32,torch.int32,torch.int64,torch.int32)):
  if not isinstance(t,torch.Tensor) or t.layout!=torch.strided or t.ndim!=1 or t.dtype!=dt or not t.is_contiguous() or t.requires_grad:raise ValueError('contiguous 1D supported integer tensor required')
  if t.device!=num_sampled.device or (require_cuda and t.device.type!='cuda'):raise ValueError('same CUDA-visible device required')
 n=num_sampled.numel()
 if not 1<=n<=32 or seq_lens.numel()!=n or idx_mapping.numel()!=n or cu_num_logits.numel()!=n+1 or prefill_len.numel()<1:raise ValueError('N1..32 length contract')
 a=num_sampled.untyped_storage();start=a.data_ptr()
 for t in vals[1:]:
  b=t.untyped_storage()
  if start<b.data_ptr()+b.nbytes() and b.data_ptr()<start+a.nbytes():raise ValueError('mutated sampled backing overlaps read-only input')
 return n
def validate_values_cpu(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len):
 vals=(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len);validate(*vals,require_cuda=False)
 if any(t.device.type!='cpu' for t in vals):raise ValueError('CPU snapshots only')
 q=cu_num_logits.tolist();ids=idx_mapping.tolist()
 if q[0]!=0 or any(a>b for a,b in zip(q,q[1:])):raise ValueError('nonnegative monotone logits prefix from0 required')
 if any(i<0 or i>=prefill_len.numel() for i in ids):raise ValueError('negative/out-of-range req index unsupported')
 if any(x<0 for x in seq_lens.tolist()+prefill_len.tolist()):raise ValueError('nonnegative lengths required')
 if any(s<0 or s>q[j+1]-q[j] for j,s in enumerate(num_sampled.tolist())):raise ValueError('sample count outside logits range')
def reference_cpu(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len):
 import torch
 validate_values_cpu(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len)
 rejected=torch.empty_like(num_sampled)
 for i,req in enumerate(idx_mapping.tolist()):
  chunked=int(seq_lens[i])<int(prefill_len[req])
  if chunked:num_sampled[i]=0;rejected[i]=0
  else:rejected[i]=cu_num_logits[i+1]-cu_num_logits[i]-num_sampled[i]
 return num_sampled,rejected
def _enqueue(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len,rejected):
 import torch,cuda.tile as ct
 from .kernels import sample_counts
 with torch.cuda.device(num_sampled.device):ct.launch(torch.cuda.current_stream(num_sampled.device),(1,1,1),sample_counts,(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len,rejected,32))
def get_num_sampled_and_rejected(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len):
 """Mutate sampled in place; return SAME sampled object and freshly allocated rejected.
 Read-only inputs preserved. No sync/fallback, current stream. Prefill lengths may
 be CUDA-visible UVA views; caller retains pinned owner through completion.
 Values must satisfy documented domain; validation does not read CUDA values.
 """
 import torch
 validate(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len)
 rejected=torch.empty_like(num_sampled)
 _enqueue(num_sampled,seq_lens,cu_num_logits,idx_mapping,prefill_len,rejected)
 return num_sampled,rejected
