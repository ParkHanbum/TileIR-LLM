import sys,os
from pathlib import Path
assert 'PYTHONPATH' not in os.environ
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from public_vllm_composed_integration_v3 import adapter,run,result_gate
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
assert len(adapter.PARTS)==7
print('Seven adapter entry import PASS, CPU only')
