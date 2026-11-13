"""Unexecuted phase1 boundary adapter; reads only bounded public table metadata."""
from .registry import Hold,require,collect_runner
class Diagnostic:
 def __init__(self):self.registry=None;self.zeroer=None;self.failed=None;self.receipt=None;self.setup_calls=0;self.setup_attempted=0;self.setup_returned=0;self.native_zero_calls=0;self.native_zero_returned=0;self.retained=[]
 def guard(self):require(self.failed is None,'latched_failure')
 def setup(self,native,runner,attention_spec_type,tensor_type):
  self.guard()
  try:
   require(self.registry is None,'duplicate_setup');self.setup_attempted+=1;ret=native(runner);self.setup_returned+=1;self.setup_calls+=1;require(ret is None,'setup_return')
   self.registry=collect_runner(runner,attention_spec_type,tensor_type);self.zeroer=runner.kv_block_zeroer
   self.retained=[runner,self.zeroer,*self.registry.strong_owners]
   require(len(self.registry.segments)==8,'observed_segment_scope')
   return ret
  except BaseException as e:
   self.failed=safe_code(e,'setup_failed');raise Hold(self.failed)from None
 def zero(self,native,zeroer,block_ids):
  self.guard()
  try:
   require(zeroer is self.zeroer and self.registry is not None,'unbound_zeroer')
   require(self.native_zero_calls==0 and len(block_ids)==32,'observed_call_scope')
   meta=zeroer._meta;require(isinstance(meta,tuple) and len(meta)==6,'meta_structure')
   tables=meta[:3];require([str(t.dtype)for t in tables]==['torch.uint64','torch.int64','torch.int64'],'table_dtypes')
   spans=[]
   for t in tables:
    require(str(t.device)==self.registry.device and tuple(t.shape)==(8,) and tuple(t.stride())==(1,) and t.storage_offset()==0 and t.untyped_storage().nbytes()==64,'table_layout')
    require(not t.is_neg() and not t.is_conj() and not t.requires_grad,'table_view_bits')
    spans.append((t.data_ptr(),t.data_ptr()+64))
   require(all(b<=c or d<=a for i,(a,b)in enumerate(spans)for c,d in spans[i+1:]),'table_overlap')
   require(all(b<=owned.base or owned.base+owned.size<=a for a,b in spans for owned in self.registry.allocations.values()),'table_owner_overlap')
   self.registry.check()
   # Future reviewed runtime only:192 bytes process-local. No raw values in receipt/errors.
   values=[t.detach().cpu().tolist()for t in tables]
   record=self.registry.compare(*values,block_ids)
   page=max(s.page//4 for s in self.registry.segments.values());block=min(1<<(page-1).bit_length(),1024)
   require(all(type(x)is int for x in meta[3:]),'launch_geometry_types')
   require(meta[3:]==((page+block-1)//block,block,8),'launch_geometry_mismatch')
   self.receipt=record
   require(record['within_budget'],'selected_union_exceeds_256MiB')
   self.native_zero_calls+=1;ret=native(zeroer,block_ids);self.native_zero_returned+=1
   require(ret is None,'native_return');self.registry.check()
   self.receipt.update(native_calls=1,original_return_preserved=True)
   return ret
  except BaseException as e:
   # Only own closed-set codes; never repr/table/address/native exception serialization.
   code=safe_code(e,'native_or_metadata_failure')
   self.failed=code;raise Hold(code)from None
 def close(self,after_synchronize=False):
  require(after_synchronize,'caller_sync_required');self.registry=None;self.zeroer=None;self.retained=[]

ALLOWED_CODES=frozenset(['ambiguous_allocation_overlap', 'block_id_bounds', 'block_ids', 'caller_sync_required', 'dedup_domain_mismatch', 'device', 'duplicate_setup', 'kernel_block_ratio', 'latched_failure', 'launch_geometry_mismatch', 'launch_geometry_types', 'logical_blocks', 'merged_extent', 'meta_structure', 'native_return', 'native_table_mismatch', 'observed_call_scope', 'observed_segment_scope', 'page_alignment', 'segment_alignment', 'segment_count_cap', 'segment_extent', 'selected_union_exceeds_256MiB', 'setup_return', 'shape', 'stale_allocation', 'stale_owner_view', 'stale_view', 'storage_origin', 'stride_alignment', 'table_count', 'table_dtype', 'table_dtypes', 'table_layout', 'table_overlap', 'table_owner_overlap', 'table_view_bits', 'unbound_zeroer', 'view_bits', 'view_extent'])
def safe_code(error,fallback):
 # Do not call str/repr on arbitrary exception or permit custom Hold subclasses.
 if type(error)is Hold and len(error.args)==1 and type(error.args[0])is str and error.args[0] in ALLOWED_CODES:return error.args[0]
 return fallback

# Closed internal binding codes; never serialize arbitrary exception messages.
ALLOWED_CODES = frozenset(ALLOWED_CODES) | frozenset(('source_changed','binding_identity','binding_signature','binding_source','constructor_alias'))
