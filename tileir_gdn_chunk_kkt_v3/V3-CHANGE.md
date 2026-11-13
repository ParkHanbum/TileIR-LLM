# Explicit K8 accumulation candidate, CPU only

Keep the RZ conversion, FP32 scaling and observed-identical exp factor unchanged. Change each logical K64 MMA into eight ascending K8 MMAs, passing the prior FP32 accumulator, for16 iterations over K128. Restrict external prior-native BK provenance to64. The aim is to constrain internal reduction association; matching native order remains a hypothesis until generated code and raw comparison verify it. No speedup claim, raw criterion change or adoption. Await operand diagnostic before selecting this candidate for GPU.
