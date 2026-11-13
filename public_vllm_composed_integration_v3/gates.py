import json,hashlib
from pathlib import Path
from packed_vllm_runtime_compare_v4.gates import public_logprobs,compare_public
P=Path(__file__).resolve().parent.parent
PROOFS=(('norm_runtime_integration_v4','normIntegration04','c095d7e62a33d0507e3d0a190e38d948ff1bd6abbe747b2bbcae1f5d2951894d'),('attention_gate_runtime_integration_v1','attentionGateIntegration01','b6121375798381b9655917c8a30c16a672d238aed7c2d52e23809721c64ad45d'),('block_gather_runtime_integration_v1','blockGatherIntegration01','5b25f22ff2bd88599941f592e97c6f023a8ab8dfe975f0a8cbc3af81575b873d'),('slot_mapping_runtime_integration_v1','slotMappingIntegration01','f43d03167f49c5b3141c6a345467c574b78fb7056fd8b309ec0cd433c071ac6b'),('pos_seq_runtime_integration_v1','posSeqIntegration01','366ca07b95b6a35f944e0681345812e8e3f161d57d561bf55fd884a35fa40930'),('gated_norm_runtime_integration_v2','gatedIntegration02','afbdd75c2994d1b1062fb5758655a935ad422af5f83263c1518c0f32b905fc75'))
def qualification(root):
 proof_file=Path(root)/'SAMPLER-FULL-INTEGRATION-PROOF.json'
 assert proof_file.is_file(),'RELEASE BLOCKED: actual samplecounts full16 proof not yet received'
 sampler=json.loads(proof_file.read_text())
 assert sampler['folder']=='sample_counts_runtime_integration_v1'
 evidence=[]
 for folder,name,expected in (*PROOFS,(sampler['folder'],sampler['run_name'],sampler['result_sha256'])):
  base=P/folder;r=base/name/'RESULT.json';l=base/(name+'-launch.json');c=base/(name+'-recovery.json')
  assert hashlib.sha256(r.read_bytes()).hexdigest()==expected
  result=json.loads(r.read_text());launch=json.loads(l.read_text());recovery=json.loads(c.read_text())
  assert result['complete'] and result['public_logprob_gate']['passed'] and launch['returncode']==0 and launch['result_gate']['passed']
  assert recovery['child_pid_absent'] and recovery['lock_available']
  if folder==sampler['folder']:
   assert hashlib.sha256(c.read_bytes()).hexdigest()==sampler['recovery_sha256']
   assert all(result[arm]['selected_dispatch']['supported_calls']==16 for arm in ('shadow_request','replacement_request'))
  evidence.append(dict(result=str(r),result_sha=expected,launch=str(l),recovery=str(c)))
 return dict(components=evidence,scope='Separate prior integrations, not evidence of composition; new1616shadow must pass')
