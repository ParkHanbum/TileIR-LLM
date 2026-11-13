# Sample-count capture01

Actual capture exit0/native512token gate and independent recovery confirmed. Firstprefill/firstdecode CPU public formula equals both native outputs exactly. N32, sampled/seq/prefix/prefill int32, reqidx int64, contiguous offset0. First-return sampled object preserved, rejected fresh/disjoint; all read-only full backing storage preserved. Both cases sampled1/rejected0, with prompt complete equality53 and decode54 versus prefill53. No chunked-prefill branch actual coverage.

Actual prefill_len is UvaBuffer pinned-host storage with same-address CUDA accelerator view, observed before/after. It must be reconstructed with public mapped-view helper and host owner retained, not silently replaced by fresh device-copy. Other inputs have captured device descriptors; public source creation path retained. No GPU TileIR claim yet. All original artifacts immutable.
