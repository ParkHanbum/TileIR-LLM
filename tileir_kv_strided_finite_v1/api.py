# SPDX-License-Identifier: Apache-2.0
"""Actual KV value layout only; not a general strided finite API."""
def all_finite_value(x):
 import torch
 from .contract import descriptor
 from tileir_validation_primitives_v3.contract import status_boolean
 assert isinstance(x,torch.Tensor) and x.device.type=='cuda' and x.layout==torch.strided
 n=descriptor(tuple(x.shape),tuple(x.stride()),str(x.dtype),x.storage_offset(),x.untyped_storage().nbytes())
 # Ordinary actual-value view only. Flagged-view transport remains unqualified.
 assert not x.is_neg() and not x.is_conj()
 import cuda.tile as ct
 from .kernels import count_rows
 from tileir_validation_primitives_v3.kernels import reduce_counts
 counts=torch.empty((n,),dtype=torch.int32,device=x.device)
 status=torch.empty((1,),dtype=torch.int32,device=x.device)
 stream=torch.cuda.current_stream(x.device)
 ct.launch(stream,(n,),count_rows,(x,counts))
 ct.launch(stream,(1,),reduce_counts,(counts,status,1<<(n-1).bit_length()))
 return status_boolean(int(status.item()),n*1024)
