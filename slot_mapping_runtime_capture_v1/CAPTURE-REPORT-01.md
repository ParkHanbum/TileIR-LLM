# Slot mapping native capture01

Actual native capture child/launcher exit0,512-token identity and independent recovery confirmed. Observed first prefill and first decode, CP1. Actual block sizes128/128/128/528, all4mapping-enabled. Output int64[4,2048],stride[2048,1],offset0. Indices int64,query starts int32,positions int64,sizes int32,enable bool. All recorded inputs preserved.

CPU public-source reference exactly reproduces both entire8192-element native post buffers. Prefill1696 actual tokens leaves1408 PAD elements across4groups; decode32 leaves8064 PAD elements. Returned prefixes retain stride2048, not compact rows. Wholecapacity PAD is part of contract. No dummy/disabled-group/CP>1 actual coverage. Device labels alone do not prove physical placement; ordinary block-table/slot allocations and the actually invoked public input preparation source are retained, with original descriptors. Pointer-table numeric addresses are provenance, not reusable pointers.

This is eager diagnostic capture, not TileIR GPU qualification or optimized-runtime equivalence. All original artifacts immutable; source/cache1033 unchanged per independent recovery. Next step two frozen input replays, raw integer/fullcapacity/input/alias checks without timing.
