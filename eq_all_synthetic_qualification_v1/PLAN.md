# EQ-all isolated same-input synthetic qualification (CPU prepared)

API: public_eq_all_guard_v2. FP32 raw load→documented uint32 bitcast, signed-zero-aware0/exact+1 predicate; no existing numerical code modified. This is190 PRIVATE synthetic cases, not actual408 model guards. NoGPU has run.

22 rawword cases include ±0, ±min/max subnormal, ±minnormal, ±Inf, multiple quiet/signaling NaN payloads, +1 and adjacent representables, negative1 and finite extrema. Each tested against constant0/1 on scalar, logical broadcast128/physical1 at storage offsets0/1, and dense1025 last element. Additional empty,1023,1024, dense2^24 and2^24 first/middle/last mismatch cases. Max owner64MiB; status scratch at most65540B excluding compiler caches. NaN payloads are constructed as signed int32 words and typed viewed as float32, never converted through Python float.

Each exact same tensor goes native public `(x==constant).all()` twice, then candidate twice; exact Python bool must match independent rawword reference and all4 outputs. Entire typed OWNER snapshot before/after EACH call, including padding outside broadcast view, must match raw bytes. Current stream must be unchanged. Final synchronization mandatory. Native GPU sNaN/denorm behavior and compiler bitcast/load preservation are unverified until actual execution; CPU tests and previous finite tests do not substitute.

Inherited isolated finite lifecycle: source/manifest pins before action, common LOCK_NB + fresh admission, one owned subprocess, entirely new per-run CUDA/cuTile/tmp directories created only during explicit actual run,585s alarm/600s parent cap, stop_child cleanup, actual exit0 + strict190-row result gate + independent post admission/lock recovery. No existing cache reused/mutated and no automatic retry. Runtime dirs/receipts are not in source preparation. Correctness only, no performance collection.

Model408 remains separate followup:384 descale checks vs24 large initial-state zero checks, same owned guard locations, finite provider and error/shortcircuit ordering preserved. Actual pointer/layout/flag/census and full model endpoints require another reviewed scope. No model fallback or100% claim here.

CPU17 tests (API12 + qualifier5) PASS. Full API/source freeze, CPU-PINS/MANIFEST and two dry launch entries are completed after coordinator GPU-window release. Framework import/GPU query/compilation/cache creation0 in preparation.
