import os,sys
from pathlib import Path
assert 'PYTHONPATH' not in os.environ
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from public_vllm_composed_integration_v9 import run,adapter,result_gate
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
assert run.final_cleanup.__module__=='public_vllm_composed_integration_v5.cleanup'
print('Attention KV integration fresh import/shared cleanup PASS; CUDA imports0')
