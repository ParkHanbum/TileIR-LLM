import os,sys
from pathlib import Path
assert 'PYTHONPATH' not in os.environ
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import pos_seq_runtime_integration_v1.adapter,pos_seq_runtime_integration_v1.run,pos_seq_runtime_integration_v1.result_gate
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
print('ENTRY_IMPORT_PASS CPU only')
