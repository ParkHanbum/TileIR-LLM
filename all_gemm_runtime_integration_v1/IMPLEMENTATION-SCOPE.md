# CPU implementation scope

Source implements the approved W-boundary addendum. Each phase scans every unique actual selected weight in <=16MiB CPU chunks, records full SHA/finite/byte count/owner mapping and compares all4 boundaries. Calls retain owner/storage/data-pointer/shape/stride/dtype/device/version checks, with inference-version absence explicit. W bytes are not checked per call.

All2,448 shadow outputs still have complete BF16 raw/strict/finite comparison. All X bytes are checked before/after native and candidate. Every replacement X hash, weight identity key and Y hash must match its verified shadow; W phase-boundary hashes also match. First numerical/API/phase failure is latched before another target/excluded entry.25 excluded apply guards must never invoke native. All153 selected paths and13 contracts are counted exactly.

Only first mismatch X/nativeY/tileY may be saved within128MiB logical cap; weights are never duplicated. CPU-side chunked comparisons cap temporary tensor sizes independently of largest GEMM. Existing model lifecycle remains900s worker/930s parent and90s compiler limit. A timeout is an incomplete proof, never a pass. No performance data is claimed.

Preparation is not GPU validation. The parent owns source review, fresh cache preseed, admission/common lock, actual execution and independent recovery. Frozen512 IDs/public2560 exact, T/T read-only, other16 non-GEMM adapters native remain mandatory.
