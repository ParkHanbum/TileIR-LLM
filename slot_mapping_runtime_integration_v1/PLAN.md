# Full16 slot mapping integration, CPU-ready

Reuse successful block-gather engine lifecycle, unchanged native precision reader, frozen512 IDs and public top5/chosen-logprob exact/tolerance gate. Only compute_slot_mappings is patched; preceding native block gather remains native. This is an independent functional port, not combined-ports or performance experiment.

Original public prepare_attn sequencing retained. Actual input-batch metadata classifies prefill/decode. Shadow16 calls: clone entire output backing BEFORE native mutation, native exactlyonce, TileIR separate output with same native input pointers, whole-capacity bytes exact. Native return object returned verbatim. CPU public reference, all input backing preservation (including dcp/pointer/stride descriptors), prefix view shape/stride/offset/alias and entirecapacityPAD checks run each time. Conditional replacement only after16 rawpasses, native slot mapper0, TileIR mutates original persistent output and returns aliasing prefix. No private binary analysis or policy change.

Scope CP1, four enabled groups, block sizes128/128/128/528, source shapes32x1/1/1/8, output4x2048. Original fullcapacityPAD semantics retained; prefix stride2048 not compact. Each arm16APIcalls=64group launches. source/runtime original tensor pointers retained; no fresh count/UVA assumption. Dummy/disabled/CP>1 remain unobserved actual scope. No ABI equivalence, optimized graph or performance claim.

Qualification3c6f95... and independent recoverya902cf... verified. CPU4tests: full16shadow/replace nativecounts and return identity, mismatch HOLD, class-boundary restore, no CUDA initialized. Cache not copied/warmed here. Parent owns GPU window and cache preseed; new sibling source only.
