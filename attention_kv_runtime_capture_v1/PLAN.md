# Native attention KV insertion capture, CPU ready

No backend forcing. Public apply_model enumerates8 actual attention modules, records impl/method and requires observed FlashAttentionImpl/imported helper; any other actual backend HOLDs before request. Outer impl and public imported helper wrappers call originals exactly once and preserve None returns. The outer unified_attention dummy is untouched.

One unobserved and one observed token-only request, same B32/P53/G16/eager/512 IDs. Observe8x16=128 actual updates. Layer3 first prefill and first decode only are snapshotted. Phase is established from actual query_start_loc differences and seq_lens:32x53/seq53 or32x1/seq54..68, not just tensor shape.

Original K/V/slots/scales are saved before/after and logical hashes checked. Cache full descriptors/alias ranges are recorded; only addressed slots are gathered before/after. No whole storage clone. Shared K/V allocation overlap is expected and not rejected. Unique nonnegative in-range slots only for this first capture; unexpected negatives/duplicates/layout/quantization HOLD without invented semantics. Snapshot individual64MiB/aggregate256MiB logical limits. These limits cover logical snapshots, not serialization overhead or total engine VRAM. Untouched cache slots/page padding are not verified. No kernel/API replacement, attention core qualification or performance claim.

Pinned original lifecycle:900s child/930s parent, common admission/lock, own process cleanup, independent recovery by coordinator. Native flags unchanged, same512 IDs, no public logprob gate claim. Source read-only operations/copy/CPU sync can affect timing, which is not used.

CPU7tests: phase/slot bounds, actual installed impl+helper wrappers full128 mock census, originalNone identity and reverse restoration, AST/noGPU import, three actual-finally lifecycle tests. Actual env PYTHONPATH unset import smoke passed. No GPU initiated.
