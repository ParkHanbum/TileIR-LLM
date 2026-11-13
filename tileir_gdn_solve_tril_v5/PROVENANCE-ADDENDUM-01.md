# License correction, additive only

The final sentence of PROVENANCE.md incorrectly said the FLA MIT notice was in LICENSE. LICENSE contains the Apache 2.0 license only. The unchanged source attribution remains applicable, but the correct license locations are now:

- LICENSE: Apache License 2.0, retained unchanged.
- LICENSE-FLA-MIT-01.txt: complete original FLA MIT license, including Copyright (c) 2023-2025, Songlin Yang, Yu Zhang, permission conditions and warranty/liability disclaimer. Copied byte-for-byte from installed vLLM 0.32.0 third_party/flash_linear_attention/LICENSE.

This addendum supersedes only the incorrect license-location statement. No numerical code, API, test, qualifier, original provenance, or prior pin manifest was modified. The kernel header reference to the original MIT attribution in LICENSE should likewise be read as LICENSE-FLA-MIT-01.txt; the header is retained byte-for-byte to preserve the reviewed numerical source.

IEEE is the native solver setting only. Candidate f32 ct.mma dtype support does not prove IEEE lowering, accumulation order or numerical compatibility. These remain subject to the planned isolated raw qualifier and public generated-code inspection.
