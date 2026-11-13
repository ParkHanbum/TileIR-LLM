# V7 narrow lifetime repair; actual endpoint pending

V6 failed with three CUDA32MiB allocation OOM warnings followed by shared-latch assertion. Its original result does not identify the first exception/allocation owner, and no lean endpoint passed. Failure proof remains separate; no automatic retry or claimed causal certainty.

Only copy adapter changes to gdn_prefill_output_copy_lean_v2. Same qualified Copy.__call__/kernel, shape/stride/address/precision/KV/settings. Public Tensor.record_stream records four typed source/destination/views on the unchanged launch stream before dropping strong pending tensor refs. Successful callback keeps no pending payload refs, so strict finish expects0 rather than24. Failed launch/record retains owners. Existing request and two cleanup sync boundaries, calls, cleanup identity and release gates remain unchanged. No per-call sync/event or rearm.

A small closed primary-failure label is stored before outer latch replacement. Exception type and selected component labels only, no repr/address/tensor dump in this new field. Existing inherited error artifacts unchanged. The observed v6 cause remains unproven beyond OOM warnings+shared latch.

CPU mocks test lifetime contract and failure retention but cannot prove actual caching allocator behavior. New endpoint/exit/recovery and same-request profile remain pending. Existing actual raw joint proof and earlier whole-v5 proof remain prerequisites, not a fabricated v6 success. GPU/performance not measured here.
