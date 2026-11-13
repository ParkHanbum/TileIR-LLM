import hashlib,json
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
def qualification(root):
 p=P/'gdn_post_conv_qualification_v2';r=p/'postConvQual02/RESULT.json';c=p/'postConvQual02-recovery.json';l=p/'postConvQual02-launch.json'
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert sha(r)=='818a7155bc29e01f249f961b1a2786a204bb929b5bd5a0c6dcd339323f333f76'
 assert sha(c)=='545c6e60aea14882023f97728aa72c2ccc68adb587907a8a29b84b51b6d08c76'
 j=json.loads(r.read_text());assert j['complete'] and len(j['records'])==4 and all(v['passed'] for v in j['records']) and json.loads(l.read_text())['returncode']==0
 rec=json.loads(c.read_text());assert rec['child_pid_absent'] and rec['lock_available']
 return dict(scope='One frozen actual prefill five-output function qualified; all24 live calls still required',result_sha256=sha(r))
