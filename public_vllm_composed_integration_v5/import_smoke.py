import os,sys
from pathlib import Path
assert 'PYTHONPATH' not in os.environ
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from public_vllm_composed_integration_v5 import run,adapter,result_gate
assert len(adapter.PARTS)==7 and 'torch' not in sys.modules and 'vllm' not in sys.modules
print('Default seven-adapter CPU import PASS')
