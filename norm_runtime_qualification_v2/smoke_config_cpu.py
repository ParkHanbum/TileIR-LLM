import os,json
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
assert not torch.cuda.is_initialized()
from vllm.config import VllmConfig,set_current_vllm_config
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
config=VllmConfig()
with set_current_vllm_config(config):
 m=GemmaRMSNorm(2560,eps=1e-6)
 assert m.weight.device.type=='cpu' and tuple(m.weight.shape)==(2560,)
assert not torch.cuda.is_initialized()
print(json.dumps({'constructor_pass':True,'forward_executed':False,'cuda_initialized':False,'config':'public default VllmConfig','compile_native':False}))
