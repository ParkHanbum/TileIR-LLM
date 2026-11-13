import json,hashlib
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 r=P/'norm_aten_order_v1/atenOrder01/RESULT.json'; c=P/'norm_aten_order_v1/atenOrder01-recovery.json'; l=P/'norm_aten_order_v1/atenOrder01-launch.json'
 assert hashlib.sha256(r.read_bytes()).hexdigest()=='6106e69cbfd0f7faad604577d41d2125c82cf4ae581a33eda9ab4b92633a01c1'
 assert hashlib.sha256(c.read_bytes()).hexdigest()=='c213ca399bdcf51cddc3c53f25d1b60739e3e7ccabceb3f8ad853dc49cab31fe'
 data=json.loads(r.read_text());rec=json.loads(c.read_text())
 assert data['complete'] and data['variants'][0]['vs_captured_native']['raw_equal'] and json.loads(l.read_text())['returncode']==0 and rec['child_pid_absent'] and rec['lock_available']
 return dict(scope='Instrumented residual first-case ATen-order diagnostic PASS only. New uninstrumented two-branch facade requires full 1040-call shadow before replacement; historical v1 4case PASS not relabelled.',result_sha256=hashlib.sha256(r.read_bytes()).hexdigest())
