# Original-v7 regression coverage

Baseline: `0b9611661ef216c35fb9f4c5da92662e5930dcbb` (`v7-own-kernel-fixed-workload-20261113`). No G/H/O/S/M or microtuning candidates.

## Historical proof, not a fresh test run

Three original-v7 requests each contain 13,300 compute launches, mapped through exact ELF symbols and pinned public source to the same 168 full symbols and 58 kernel files. This is a compute-kernel denominator, not every runtime operation. The mapping is not a loaded-module SHA observation. Original-v7 endpoint proof checks 512 token IDs and 2,560 public logprob values. Separate historical GEMM raw proof covers 13 contracts / 2,448 calls; this is not universal cuBLAS equivalence.

| Functional family | Kernel files | Full symbols | Launches/request |
|---|---:|---:|---:|
| Packed and unpacked projection GEMM | 4 | 11 | 2448 |
| Gated delta network prefill/decode and state | 10 | 13 | 1753 |
| Causal convolution and post-update state | 4 | 5 | 448 |
| Attention and KV cache | 5 | 6 | 512 |
| Normalization, RoPE, activation, embeddings | 7 | 14 | 4240 |
| Sampling and public logprob outputs | 10 | 25 | 400 |
| Finite/range/equality and ownership guards | 6 | 43 | 2964 |
| Runner metadata, typed copies and initialization | 12 | 51 | 535 |

## Executable representative regression matrix

Current CLI provides 10 module groups. Each row has independent CPU tests and an
explicit GPU entry; actual PASS requires external receipts for the exact full
HEAD. The previous a8ffb8b milestone passed its 7 groups and whole replay; it does
not automatically qualify these later sources. New module/whole runs and their
recovery are recorded outside Git, without making a commit claim about itself.

| Group | Dedicated fixture and scope |
|---|---|
| gemm | 3 actual saved matrices, fresh D99 references; not all 13 contracts/2448 calls |
| predicates | 20 synthetic finite/equality cases; not all guard APIs |
| gdn | 2 actual decode banks, compact 33-slot owner, native/v7 4 records |
| norm_conv | 2 actual prefill norms and 1 decode conv, native/v7 6 records; minimum-span state |
| attention_kv | 6 actual attention banks x4 and 2 KV banks x4; reconstructed/sentinel KV |
| gdn_prefill | actual layer0, 9 stages/output/state x4 plus 3 synthetic parent fixtures x4; not original full owner |
| sampling_runtime | actual calls0/15 greedy/top5/logprob/rank, selected-row embedding, synthetic metadata/cast; 16 records |
| pointwise_remaining | actual prefill/decode QK/RoPE shared storage and SiLU inputs, native/v7 x2; 16 records |
| recurrent_remaining | 6 actual gated/norm/prefill-conv/post-conv boundaries, native/v7; 12 records |
| runtime_remaining | 17 bounded synthetic constructors/fill/NOT/scatter/cast/slot/pos/staged/zero contracts; not live UVA/registry |

Whole proof checks fixed 512 IDs/2560 public values, dispatch, same-request own
compute census, classic/Lt positive control and no lean compute calls, actual
exit/recovery and unchanged own-artifact local-call evidence. It does not replay
every intermediate raw tensor or independently test every 58 source files.

Small boundaries without dedicated fresh GPU cases in these suites include
GDN internal copies/prepare/output-copy, attention-gate/KV finite helpers,
ownership-pair EQ/range/temperature/post-update guards, sampler counts/cat glue,
block gather and residual RoPE combine/bias/post-update glue. Their current
coverage is whole E2E plus selected CPU contract/lifetime tests and separately
identified historical proofs. Live owner registration/UVA retirement remains
E2E scope even where private staged/zero leaves have independent fixtures.

The immutable baseline tag is provenance, not a standalone runnable checkout.
Unchanged lazy facade/import support and historical negative fixtures were added
in later regression commits. Reproduce using the exact tested HEAD; never move
the tag. Future source changes follow WORKFLOW.md and its separate performance
gate; no performance result is created by this matrix.

## Error, lifetime and compatibility boundaries

- `whole_lean_extensions_runtime_v7.test_failure_cpu` and `.test_cpu` are existing whole-path CPU hooks; they are run by the isolated CPU CLI, not by reading this catalog. GDN lifetime and metadata/ownership modules are listed separately in JSON. Negative inputs, first-failure latching, foreign-binding preservation and owner retirement must remain explicit tests, not inferred from correct numeric outputs.
- Typed integer-finite paths use metadata plus native stream synchronization, not a TileIR kernel. Host/device transport, allocations and reference lifetime remain runtime obligations.
- APIs are functional Python adapters for the fixed eager workload, not cuBLAS ABI replacements, arbitrary models/shapes or CUDA-graph compatibility.
- Saved tensors, model/checkpoint, native libraries, runtime installation and historical evidence are external hash-bound inputs. A source-only checkout is not self-contained.
- Raw bits, numerical tolerance, public model outputs, functional coverage and speed are separate verdicts. Any fresh PASS is exact-HEAD and fixture-scoped; no performance-equivalence claim is made.

`catalog.json` contains the exact 58-source grouping, historical artifact hashes, and callable CPU/GPU module matrix. Historical manifests remain immutable; new execution receipts belong outside the checkout.
