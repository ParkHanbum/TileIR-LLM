import os,sys
from pathlib import Path
assert 'PYTHONPATH' not in os.environ
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from public_vllm_composed_integration_v6 import run,adapter,result_gate
assert len(adapter.PARTS)==9 and 'torch' not in sys.modules and 'vllm' not in sys.modules
print('Default nine-adapter CPU import PASS')
