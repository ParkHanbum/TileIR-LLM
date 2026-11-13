"""Public runner activation row gather, inference-only, no native fallback."""
def validate_metadata(ids,weight,values):
 if ids['shape']!=(32,) or ids['stride']!=(1,) or ids['dtype']!='torch.int64':raise ValueError('1D actual workload integer IDs only')
 if weight['shape'] not in ((1696,2560),(32,2560)) or weight['stride']!=(2560,1) or weight['dtype']!='torch.bfloat16':raise ValueError('unquantized contiguous BF16 table only')
 if ids['offset']!=0 or weight['offset']!=0:raise ValueError('offset')
 if len(values)!=ids['shape'][0] or any(type(v)is not int or not 0<=v<weight['shape'][0] for v in values):raise ValueError('ID bounds')
 return (len(values),2560)

def gather_hidden_rows(weight,ids):
 """Fresh contiguous BF16 rows, bit-preserving. Host bounds check synchronizes.
 Caller must not concurrently mutate input IDs/table. No backward, quantization,
 TP>1, negative indices or other shapes; no throughput claim.
 Repeated and reordered in-range indices are supported; no decode alias shortcut.
 """
 import torch
 def meta(t):
  if not isinstance(t,torch.Tensor) or t.layout!=torch.strided or t.device.type!='cuda' or t.device!=ids.device or t.requires_grad or t.is_conj() or t.is_neg():raise ValueError('same CUDA, non-grad strided tensors')
  return dict(shape=tuple(t.shape),stride=tuple(t.stride()),offset=t.storage_offset(),dtype=str(t.dtype))
 im,wm=meta(ids),meta(weight)
 shape=validate_metadata(im,wm,ids.detach().cpu().tolist())
 a,b=ids.untyped_storage(),weight.untyped_storage()
 if max(a.data_ptr(),b.data_ptr())<min(a.data_ptr()+a.nbytes(),b.data_ptr()+b.nbytes()):raise ValueError('input owner overlap')
 # Reuse the already preserved public-source raw row-copy kernel, no new math.
 from tileir_embedding_rope_v1.kernels import embedding_raw_kernel
 import cuda.tile as ct
 with torch.cuda.device(ids.device):
  out=torch.empty(shape,dtype=torch.bfloat16,device=ids.device)
  n=shape[0]*shape[1]
  ct.launch(torch.cuda.current_stream(ids.device),((n+1023)//1024,1,1),embedding_raw_kernel,(weight.view(torch.int16).view(-1),ids,out.view(torch.int16).view(-1),n,2560,1024))
 return out
