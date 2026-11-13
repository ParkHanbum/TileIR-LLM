"""Compose sixteen independently qualified, public-source functional adapters."""
from pathlib import Path
from norm_runtime_integration_v4 import adapter as norm
from attention_gate_runtime_integration_v1 import adapter as attention
from block_gather_runtime_integration_v1 import adapter as blocks
from slot_mapping_runtime_integration_v1 import adapter as slots
from pos_seq_runtime_integration_v1 import adapter as positions
from gated_norm_runtime_integration_v2 import adapter as gated
from sample_counts_runtime_integration_v1 import adapter as samples
from qk_rope_runtime_integration_v1 import adapter as qk
from silu_runtime_integration_v2 import adapter as silu
from indexed_decode_conv_runtime_v1 import adapter as conv
from indexed_prefill_conv_runtime_v3 import adapter as prefill
from gdn_post_conv_runtime_integration_v2 import adapter as postconv
from tp1_embedding_runtime_integration_v1 import adapter as embedding
from attention_kv_runtime_integration_v2 import adapter as kv
from greedy_sampler_runtime_integration_v1 import adapter as greedy
from gdn_decode_core_runtime_v4 import adapter as gdn_core
ATTR='_tileir_public_composed_v11'
PARTS=(('norm',norm,1040),('attention_gate',attention,128),('block_gather',blocks,16),('slot_mapping',slots,16),('pos_seq',positions,16),('gated_norm',gated,384),('sample_counts',samples,16),('qk_rope',qk,128),('silu_mul',silu,512),('indexed_prefill_conv',prefill,24),('indexed_decode_conv',conv,360),('post_conv',postconv,24),('embedding',embedding,16),('attention_kv',kv,128),('greedy_sample',greedy,16),('gdn_decode_core',gdn_core,360))
def install(model,output):
 assert not hasattr(model,ATTR)
 root=Path(output);root.mkdir(exist_ok=False);installed=[];results={}
 setattr(model,ATTR,installed)
 try:
  for key,module,count in PARTS:
   results[key]=module.install(model,str(root/key));installed.append((key,module))
  return dict(components=results,logical_calls_expected=3184,scope='Sixteen functional public-source adapters; no combined optimized graph or ABI claim')
 except BaseException:
  close(model);raise
def start(model,mode):
 assert len(getattr(model,ATTR))==16
 return {key:module.start(model,mode) for key,module,count in PARTS}
def finish(model):
 components={key:module.finish(model) for key,module,count in PARTS}
 for key,module,count in PARTS:assert components[key]['supported_calls']==count
 return dict(components=components,shadow_pass=all(v['shadow_pass'] for v in components.values()),supported_calls=sum(v['supported_calls'] for v in components.values()),supported_vendor_fallback_calls=sum(v.get('supported_vendor_fallback_calls',0) for v in components.values()),convolution_scope='decode prefill counter24 is delegation to prefill TileIR adapter in replace, not native execution; prefill update identity differs while decode wrapper is active')
def close(model):
 installed=getattr(model,ATTR,None)
 if installed is None:return dict(restored=True)
 results={};errors=[]
 for key,module in reversed(installed):
  try:results[key]=module.close(model)
  except BaseException as e:errors.append((key,repr(e)))
 delattr(model,ATTR)
 assert not errors,errors
 assert all(v['restored'] for v in results.values())
 return dict(restored=True,components=results)
