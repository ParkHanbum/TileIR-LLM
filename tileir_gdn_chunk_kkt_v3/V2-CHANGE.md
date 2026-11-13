# Supported RZ conversion candidate

v1 actual compile failed before any candidate output: f32 bitwidth32 and tf32 bitwidth19 cannot be bitcast. No numerical mismatch was observed. Original failure and v1 source are preserved. v2 changes only two operand conversion expressions to astype(tfloat32, rounding_mode=RZ). Public cuda.tile astype documents explicit float rounding; RZ means towards zero. The installed public conversion registry explicitly permits F32→TF32 RZ at TileIR13.4+, matching installed tileiras13.4.92. This uses supported conversion and does not bypass verifier/type checks.

RZ is a testable truncation hypothesis, not proven equivalence to native TF32 instruction behavior. The previous native reference and actual stable BK64/warps8/stages3 are preserved from the failed-candidate run. Candidate compilation, numerical equality and performance remain unverified. Remaining accumulation and exp differences may still exist.
