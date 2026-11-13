# Ordinary packed recurrent candidate, NOT yet qualified
Only captured B32,Hk16,Hv32,K/V128,BF16 packedinputs,out andFP32state/A_log. State[slot,Hv,V,K] withdenseinneraxes,actualpagepitch supported. Caller out/initialstate updatedinplace,returns exactlysameobjects. Slots<=0 zerooutput/no stateupdate;positiveunique slots,bounds checkedonCPU (synchronizes). Bias/gating/internalparameters fixedpublicsource:QKnormeps1e-6 sqrtapproxFTZ/full division,scale128^-0.5,softplusthreshold20,FP32beta,decay/delta/newstate/outputorder. No denseTileGymkernel reuse/no transpose trick. No vendorfallback/backward/speculation/varlen/prefill.

Public source vLLM0.32 fused_recurrent.py (Apache2.0/FLA); existing public Triton generated TTGIR #blocked[4]/32threads and[1,4]/[1,32] mapsfourconsecutiveK perlane. Source-order explicit4localterms then butterfly16,8,4,2,1 implemented. Candidate scalar log vsnativecompiled __nv_logf and FMA contraction/lowering remain numerical risk, not assumed equivalent. Wholeactualbank raw+strict+finite controls mustpassbeforeanysupportclaim. No arbitrary overflow/subnormal guarantees, no performance claim. Source-generated PTX evidence isnotvendorbinaryanalysis.

## Version2 narrow hypothesis
Only the four component state-update expressions are reordered from h_decay+delta*k to k*delta+h_decay. No norm, gate, softplus, log, reduction, shape, dtype, alias, or API change. Public compiler's lhs-first fusion pattern and external-use rejection motivate this. No explicit Python FMA API exists in the inspected cuda.tile1.7; no guarantee of fusion, bit equality or support until compiled/qualified. No FP64, correction, private hook, compiler-source modification.

## Version3 narrow decay-dot order hypothesis
Only initial local dot expression reversed to h0*k0+h1*k1, so lhs-first fusion can round h1*k1 then fuse h0*k0, matching public native PTX. Stateupdatev2 retained. Output-dot/norm/gates/log unchanged. No GPU qualification or universal equality claim.

## APIv4 pending qualification
Only initial local outputdot expression reverses to h0*q0+h1*q1 to enable product1 then FMA0. State/norm/gates remain unchanged. General public rounding-order hypothesis, no bitcorrection.
