# Native block gather capture 01

Actual child/parent exit 0 and independent recovery are linked below. One warmup plus one observed B32/P53/G16 native eager request reproduced all 512 frozen token IDs. Observer saw one prefill and fifteen decode boundaries; only the first of each was saved. Compilation and graphs were disabled: not optimized-runtime equivalence.

Actual index dtype is torch.int64, counts int32 [4,32], source/output three [32,1] groups and one [32,8] group. Actual/padded request count both32, so no padded-row coverage. Both native post buffers exactly match the CPU public-source reference over all capacity elements, including unwritten tails. All recorded input backing storage preserved.

Counts use actual UvaBuffer pinned-host storage with CUDA accelerator view; CPU and accelerator pointers coincide. This is not fresh CUDA allocation. Planned TileIR qualification must retain mapped placement as primary and explicitly label a fresh device-copy control. No TileIR GPU correctness claim yet.

Initial launcher invocation used unsupported CLI flags and exited before lock/child/artifact creation; corrected positional invocation ran once successfully. Recovery preserves that history. Existing run evidence/source unchanged. CPU artifact hash verification and reference reconstruction initialize no CUDA.
