import json,hashlib
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 rpath=P/'norm_runtime_qualification_v2/normQual01/RESULT.json';lpath=P/'norm_runtime_qualification_v2/normQual01-launch.json';cpath=P/'norm_runtime_qualification_v2/normQual01-recovery.json'
 assert hashlib.sha256(rpath.read_bytes()).hexdigest()=='e447d80f8e6abcaa6628c275a7bb01c7bcca73730b5fa5968adf2b824b2facff'
 assert hashlib.sha256(cpath.read_bytes()).hexdigest()=='8246736a0876957b58499fd35123e885a24b453399adf1cd266302f6e3a92253'
 r=json.loads(rpath.read_text());l=json.loads(lpath.read_text());c=json.loads(cpath.read_text())
 assert r['complete'] and r['gate']['raw_pass'] and l['returncode']==0 and c['child_pid_absent'] and c['lock_available']
 return dict(scope='4boundary standalone norm qualification prerequisite, not all-module proof',result_path=str(rpath),result_sha256=hashlib.sha256(rpath.read_bytes()).hexdigest())
