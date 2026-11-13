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

## Fresh regression matrix: all pending

The catalog lists existing CPU modules as reusable entry points, not newly passed tests. CPU runner must execute suites in isolated CPU-only processes. GPU entry points never run automatically; coordinator approval/admission and independent cleanup evidence remain required.

- **gemm**: 3 saved matrices: layer1 ba decode, qkvz decode, gate/up prefill; not all 13 contracts or all 2448 calls. Status: pending execution receipt.
- **gdn**: 2 saved actual banks, native and original decode-v4; compact 33-slot remap, not full original backing or GDN prefill. Status: pending execution receipt.
- **predicates**: 20 private finite/equality cases; not all predicate APIs or all workload inputs. Status: pending execution receipt.
- **whole**: planned original-v7 fixed-workload endpoint, dispatch, cleanup and fresh trace audit; no optimized variants. Status: pending runner and execution receipt.

Other functional families currently have historical source/endpoint evidence and selected reusable CPU tests, but no dedicated fresh GPU regression entry. A small GEMM or predicate sample cannot certify all 58 kernel files.

## Error, lifetime and compatibility boundaries

- `whole_lean_extensions_runtime_v7.test_failure_cpu` and `.test_cpu` are existing whole-path CPU hooks; they are not run by this catalog. GDN lifetime and metadata/ownership modules are listed separately in JSON. Negative inputs, first-failure latching, foreign-binding preservation and owner retirement must remain explicit tests, not inferred from correct numeric outputs.
- Typed integer-finite paths use metadata plus native stream synchronization, not a TileIR kernel. Host/device transport, allocations and reference lifetime remain runtime obligations.
- APIs are functional Python adapters for the fixed eager workload, not cuBLAS ABI replacements, arbitrary models/shapes or CUDA-graph compatibility.
- Saved tensors, model/checkpoint, native libraries, runtime installation and historical evidence are external hash-bound inputs. A source-only checkout is not self-contained.
- Raw bits, numerical tolerance, public model outputs, functional coverage and speed are separate verdicts. No fresh all-module PASS or performance-equivalence claim is made.

`catalog.json` contains the exact 58-source grouping, historical artifact hashes, and callable CPU/GPU module matrix. Historical manifests remain immutable; new execution receipts belong outside the checkout.
