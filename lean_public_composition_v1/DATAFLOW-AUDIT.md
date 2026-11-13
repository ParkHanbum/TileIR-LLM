# Dataflow and diagnostic separation audit

All old sources/proofs remain referenced, not edited. No numerical kernel was copied or rewritten. `sampler.build_copy` is the already qualified public helper reconstruction, used unchanged. New files contain bindings only.

| Binding family | Omitted proof-only work | Retained dataflow / scope contract |
|---|---|---|
| Gemma and ordered gated norms | native shadow, reference comparisons, snapshots, hashes, per-call writes | original residual argument, detached same-storage X/Z/W views, same public API output/tuple; exact owner/width/activation/phase |
| Attention gate | native gate reference and gate/output diagnostics | same original qkv projection, fused QK/RoPE call, native attention core and output projection order; same gate API |
| QK/RoPE | comparison clones/hashes | original project wrapper, same qkv storage, original positions/weight/cache association, returned Q/K/gate identities and V view offset/stride |
| SiLU and embedding | input/output snapshots and native shadow | caller-owned SiLU output view, no replacement output allocation in binding; original embedding IDs/W and same candidate API |
| All GEMM | W byte hashes at phase boundaries, X/Y byte hashes and native comparisons | every call owner/q/weight identity, metadata/storage/version, exact shape/stride/no-bias/dispatch; K128 only ba decode; same public API allocation and returned Y |
| Gather | full table oracle/sentinel copies | original source tables, same live output tables, actual pinned UVA count owner/pointers held alive |
| Slot mapping | full output oracle/sentinel comparisons | original output buffer, CP1, exact block-size/all-enabled small value guards |
| Positions | output prefix/tail proof copies | same input/output buffers, bounded nonnegative index/start/computed-token value guards; original alias mutations |
| KV scatter | selected cache snapshots and before/after proof bytes | original live key/value cache views, original slot/scales association, finite input and phase guards; API mutations preserved |
| Prefill/decode convolution | shadow state/backing clones, holes/output comparisons | original X and state through detached same-storage views; prefill53/allFalse scope; decode wrapper delegates to earlier installed TileIR prefill; no native selected fallback |
| Post-conv | five-output reference copies/comparisons | unchanged five-output API, same input views/flags and finite scope |
| GDN decode | selected-state/backing proof copies | original state/out storage and exact caller return objects, positive unique32-slot scope, original scalar flags |
| Sample counts | copies/hash/sentinel comparison | same sampled-count object, same five views, live pinned UVA owner retention |
| Greedy | native argmax/histogram comparison | original no-drafting/temperature-zero scope and unchanged public finite/input checks |
| Top5/ranks/logprob | external raw and consistency proof comparisons | exact qualified three-API helper copy; native cat and output construction stay necessary |

`detach()` changes Tensor/autograd wrapper identity but not data storage. Where the original caller requires exact object identity (GDN returned out/state, sample counts first result), the wrapper returns/checks the original object rather than a diagnostic copy. Public APIs and native wrappers remain responsible for their original output allocation, Tensor views and in-place mutation. Their code and source hashes are pinned.

Small `.cpu()` reads and finite/value checks in metadata, convolution, recurrent and sampler wrappers are retained specifically because deleting them would expand the qualified input domain. They can incur synchronization/native kernels. Per-call metadata/count checks and original source identity checks also remain. Thus dispatch counts are not the total GPU-kernel denominator and cannot establish a TileIR percentage.

Shared bindings: prepare_attn nests gather then slots; sampler __call__ nests counts then greedy then three-way; convolution fn nests prefill then decode delegation; QK class/global helper and attention gate instance forward coexist; embedding quant_method.embedding is separate from GEMM quant_method.apply. Reverse close removes later bindings first. CPU tests exercise the actual shared wrapper functions, not a string substitution simulation.

Dropped guarantee in lean execution: there is no W full-byte boundary check or per-call X/Y raw/hash evidence. The historic combined proof retains its four W boundaries and complete raw checks. New end-to-end qualification will compare all frozen IDs/public logprobs and dispatch census against a same-runtime native baseline; it is not a second per-intermediate raw proof. If an output mismatch occurs, stop and return to the unchanged proof adapters to localize it, as a separately authorized run.
