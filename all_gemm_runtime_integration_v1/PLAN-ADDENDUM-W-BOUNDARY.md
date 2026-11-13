# Approved verification scope change before implementation

This addendum supersedes only the per-call weight-byte hashing cost paragraph in PLAN.md. The original plan and stronger ba360 experiment remain unchanged.

For every unique actual weight tensor, compute full logical/dense backing SHA256 at the start and end of each phase. Shadow-start, shadow-end, replacement-start and replacement-end must all match. Every call also checks the same owner/weight identity, shape, stride, dtype, device, storage pointer/span/offset, data pointer and version when available. Inference tensors without a version counter are explicitly labeled unsupported for that check. Weight metadata checks do not prove byte immutability between phase boundaries; no per-call W byte-preservation claim is made.

Every call still preserves all X bytes, compares original/candidate Y with unchanged raw/strict/finite gates during shadow, and links replacement X/Y hashes with the corresponding verified shadow call. Replacement weight identity/metadata and phase-boundary full weight hashes connect the W values. No numeric tolerance, native precision setting, input/output criterion or user raw-equality requirement is relaxed.

This removes repeated scans of invariant shared weights: roughly4 full selected-weight scans rather than80 (unique-storage deduplication can reduce it further). Exact scan coverage and bytes are recorded, never merely assumed from version counters. A first failure is latched and prevents every subsequent instrumented target or excluded apply from entering native or candidate work. Bounded cleanup/restoration still runs. No GPU/model/compile/cache action is authorized by preparation alone.
