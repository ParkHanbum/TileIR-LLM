from pathlib import Path
import hashlib,json,sys

def source_preflight(root):
    root=Path(root)
    expected='/home/n/Documents/Codex/2026-10-28/task-2/qwen38-vllm-ir-20261029/venv/bin/python'
    assert str(Path(sys.executable).absolute())==expected,'Use actual runtime venv Python for AST/source validation'
    receipt=json.loads((root/'CPU-RUNTIME.json').read_text())
    assert receipt['python_version']==sys.version
    pins=json.loads((root/'PINS.json').read_text())
    for path,digest in pins.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,('source changed',path)
