"""Reuse prior full component proofs; they do not qualify default combined inputs."""
from pathlib import Path
from public_vllm_composed_integration_v3.gates import public_logprobs,compare_public,qualification as previous_qualification
def qualification(root):
 r=previous_qualification(Path(__file__).resolve().parent.parent/'public_vllm_composed_integration_v3')
 r['scope']='Prior component proofs only; new seven-adapter True/True1616 shadow required, no F-result relabelling'
 return r
