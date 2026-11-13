# Narrow corrections to frozen v2
1. Record gpu_executed=True and finite_edge_probe phase durably before calling the edge probe. On probe failure, engine_init is never reached.
2. Require v1 finiteModel01 result, launch and independent recovery byte hashes, original v1 audit, actual exit0, cleanup and common result association before admission.
No numeric/API/binding/census/error-order changes. Old v2 remains frozen. No cache copied. Full pin validation pending until parent releases GPU window.
