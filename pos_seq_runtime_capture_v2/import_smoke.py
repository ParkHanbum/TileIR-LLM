"""Run with the real child Python/cwd and no PYTHONPATH; no GPU imports."""
import os,sys
from pathlib import Path
assert 'PYTHONPATH' not in os.environ
assert Path.cwd()==Path(__file__).resolve().parent.parent
import observer,run,result_gate
assert 'torch' not in sys.modules and 'vllm' not in sys.modules and 'cuda' not in sys.modules
print('ENTRY_IMPORT_PASS: observer/run/result_gate; no Torch/vLLM/CUDA import')
