"""CPU-only preparation contract. No runtime adapter imports or GPU entrypoint."""
EXPECTED={'norm':1040,'attention_gate':128,'block_gather':16,'slot_mapping':16,'pos_seq':16,'gated_norm':384,'sample_counts':16}
ORDER=tuple(EXPECTED)
TOTAL=sum(EXPECTED.values())
def require_full_evidence(proofs):
 if set(proofs)!=set(EXPECTED):raise ValueError('All seven full-integration proofs required; representative samples insufficient')
 for key,count in EXPECTED.items():
  p=proofs[key]
  if p.get('logical_calls')!=count or p.get('actual_exit')!=0 or p.get('full_integration_pass') is not True or p.get('independent_recovery') is not True:raise ValueError('Incomplete full-integration evidence: '+key)
 return True
