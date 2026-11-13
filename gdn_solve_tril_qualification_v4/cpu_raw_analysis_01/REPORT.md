# Stage3 v4 HOLD: all remaining mismatches are signed zero

All3,473,408 output elements checked. Exactly17,120 mismatches, each native +0 versus candidate -0; no normal/subnormal nonzero difference. All inside valid lower triangle. Native2 raw exact, candidate1 unrun, raw HOLD preserved.

Own generated PTX confirms512 FMA,24 +0-initialized16-step chains and8 chains whose initial accumulator equals a preceding dot result, matching the observed native carry class. Register edges are saved, not only counts. This does not establish full per-lane operand mapping, but the nonzero differences from v3 disappeared on this bank. No TF32 or explicit FTZ.

Native public PTX maps unary negation of block dot to positive-zero subtraction: mov.b32 r394,0f00000000 then12 sub.f32 instructions at the six offdiagonal result boundaries. Own uses neg.f32. IEEE RN positive0 minus positive0 yields positive0, while sign negation yields negative0. This is a concrete original operation boundary, not a data-specific output canonicalization. Native recurrence also has8 FMA versus own separate recurrence arithmetic, but all diagonal-block BF16 outputs already match. That remaining recurrence difference is not proven irrelevant to arbitrary inputs; do not change it simultaneously.

Next minimal proposal: at only six x21/x32/x43/x31/x42/x41 post-dot negation boundaries, replace unary minus with ct.sub(zero,dot,rounding_mode=RN,flush_to_zero=False). Do not inspect output values or force zeros. Preserve recurrence and every other expression. Verify own generated sub(+0,dot) and raw qualifier before adoption. No compiler/GPU run during analysis, no tolerance relaxation or whole-core/model claim.
