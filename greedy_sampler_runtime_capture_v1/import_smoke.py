import os,sys,importlib.util
from pathlib import Path
assert 'PYTHONPATH' not in os.environ
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run,launch,observer,result_gate
from preflight import preflight
preflight(Path(__file__).resolve().parent)
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
print('Fresh entry import/preflight PASS; GPU imports0')
