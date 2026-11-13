# Live ba decode K128 integration: correctness only

Prerequisite: the saved5-matrix K128 qualifier passed under fresh D99/math0 and original T/T flags. This preparation does not execute GPU or compile. Independent review and parent-controlled fresh admission/common lock are required; native cache is separately copied by the parent, never moved/deleted.

Only24 `UnquantizedLinearMethod.apply` instance bindings whose module is `layers.i.linear_attn.in_proj_ba` are temporarily wrapped. Their selected public `_gemm_impl` must be the original `default_unquantized_gemm`. Per request each owner gets1 prefill call (1696x2560, always native) and15 decode calls (32x2560,64 output channels). Other GEMM bindings remain untouched and are checked unchanged. Unexpected owners, shapes, bias, storage span or shared method instances fail closed, not fallback.

First request is all-native trajectory with360 decode comparisons. Each original apply runs exactly once and its original output object is returned. Same X/W bytes feed the new K128 candidate. Full dense X/W backing hashes, detach pointer/stride/offset identity, output contract and raw/strict/finite comparison must pass. First mismatch saves small X/W/nativeY/tileY tensors and stops; no replacement runs. Every target module has exactly15 decode/1prefill coverage checked.

Only after the complete360 shadow passes, the second request replaces360 decode applies without calling their native GEMM. It still executes24 native ba prefill calls. Each replacement must see the same X/W hashes and produce the same output hash as the corresponding verified native-shadow call, as well as passing storage/alias/finite checks. Thus raw output checks are not inferred merely from token agreement. All other GEMMs remain native; supported vendor fallback count must be0.

Both requests must match fixed512 token IDs and exactly2,560 public top5/chosen logprob values. Original True/True flags are read and preserved; no math-mode setters are used. Eager execution/compilation NONE/graph NONE with original async scheduling, same53-token x32/16generation workload. No performance claim; inherited timestamps are diagnostics only.

The reused model lifecycle has900s worker/930s parent bound, best-effort restoration and shutdown, real exit and independent recovery. Scope is only ba decode360, not all GEMMs or100% workload support. Source and all prerequisite results are pinned; the old sampled failures remain intact.
