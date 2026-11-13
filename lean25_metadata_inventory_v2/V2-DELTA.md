# V2 lifetime correction (CPU draft, not GPU release)

V1 is preserved, unrun, and HOLD: module wrappers installed after lean owned-copy construction could not reach copied `prepare_inputs` globals.

V2 installs request helper wrappers after the native reference ends and before lean adapters construct copied bodies. Wrappers remain installed until the existing numerical adapter cleanup and engine shutdown have been attempted, then restore in reverse order. The ledger is active only inside the observed lean execution: install/finish/cleanup calls delegate directly to the exact original callable without metadata or census. Engine setup wrappers remain separate.

Request metadata is preliminary until final cleanup. Final METADATA and RESULT are committed after helper restoration/census. Any adapter/helper restoration or metadata finalization failure sets complete=false. Existing numerical cleanup, kernels, value guards, precision and KV behavior are unchanged. No source cache has been created.

The stdlib regression compiles the installed public prepare_inputs and invokes actual lean owned_copy, reaches both prefill and combine helper aliases through the unmodified copied body, checks exactly-once originals and return identity, and restores bindings after a deliberate exception. Three additional cleanup cases verify success, adapter failure, and helper failure still attempt both cleanup layers and final failure-preserving save.

Metadata-only limitations from V1 remain: no tensor or scalar input payload; no pointer-table values; no general active-lane, duplicate-index, raw-address owner or lifetime proof. Diagnostic only, not performance.
