"""Closed diagnostic labels only. Never persist repr, addresses, or tensor data."""
KINDS=frozenset(('OutOfMemoryError','MemoryError','RuntimeError','AssertionError','ValueError','TypeError','KeyError','TimeoutError','NotImplementedError'))
COMPONENTS=frozenset(('output_copy/entry','output_copy/install','post_guard/entry','post_update_guard','new_domains/finish','start','install','predicate/install','predicate/start','predicate/finish','gdn_prefill_core','gdn_decode_core','all_gemm','attention_core','attention_kv','sampler_four','greedy_sample','metadata_batch/entry','scalar/entry','zero/start','zero/finish','range48-lean','sampling96-lean','temperature16-lean','finite504','eq408'))
def record(component,error):
 kind=type(error).__name__
 return dict(component=component if type(component)is str and component in COMPONENTS else 'other',error_type=kind if kind in KINDS else 'OtherException',out_of_memory=kind in ('OutOfMemoryError','MemoryError'))
def snapshot(ext):
 if ext is None:return dict(available=False)
 c=ext.controller
 primary=getattr(c,'primary_failure',None)if c is not None else None
 b=ext.bus.failure
 bus=None if b is None else dict(component=b.get('component')if b.get('component')in COMPONENTS else 'other',error_type=b.get('error_type')if b.get('error_type')in KINDS else 'OtherException')
 return dict(available=True,controller_primary=primary,extension_bus=bus,copy_failed=bool(ext.output_copy is not None and ext.output_copy.failed),pair_failed=bool(ext.ownership_pair is not None and ext.ownership_pair.failed))
