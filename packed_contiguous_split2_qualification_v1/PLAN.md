# Five-matrix D99 qualifier, CPU preparation only

No new GPU, query, lock or compilation executed by the author. Lead owns the GPU window.
Entry: launch.py --manifest MANIFEST.json --manifest-sha <SHA> --run-name splitTwo01.
Without --run-gpu it only validates pins and prints a command; add --run-gpu only after review/admission.

The two previously frozen candidate arms are separate hypotheses. For each saved matrix, first repeat public D99/math0 twice, check against stored default native except the explicitly F-trajectory layer1 bank, and reproduce stored original TileIR output twice. Default native comparison never switches to F. Torch flags are read-only and must remain True/True. No model load or replacement, no timing.

Then each still-active arm runs at most twice. First raw/strict/finite failure immediately stops that arm; it is never tried on subsequent matrices. The other predeclared arm can continue. If both fail, the run ends HOLD/actual exit2. An arm passes this *five-matrix qualifier only* if all five matrices and both repeats match default native raw/strict/finite. Controls, reference drift, input mutation, compiler/resource failure or timeout stop the entire job with actual exit1. No silently skipped regression becomes a pass. Reference repeat uses one independent public handle and current stream, not the original runtime handle/workspace.

Case order: ba decode; ba prefill; qkvz decode; gate_up prefill; first-F layer1 ba. Each owns exact saved X/W and original strided offset restoration. Full backing including prefix/holes is compared, outputs must be fresh contiguous BF16. Saved F-native output on the last matrix is a labeled historical comparison, never the target. The original baseline is replayed, but previous failed round-robin partial2/4 are not.

Public loaded cuBLAS identity, HOST pointer mode, alpha1/beta0, BF16/compute32F, algorithm99 and math0 are recorded. The existing public C wrapper is used only for create/destroy; its default-algo -1 callable is not used. Source and saved artifact hashes are pinned, caches are run-local, 90s per compiler and 570s worker/600s parent bounds. At most two failed full output files, each <=64MiB; no new full matrix captures. A successful arm would still need all eight fixtures and default-native1408-call/model checks before adoption.

Launcher derives from the reviewed GDN cumsum isolated launcher, retaining common flock, fresh admission, real child exit, stop_child cleanup, post-probe and lock reacquisition. Lead must do an independent recovery check after execution, just as other windows. This qualifier has no automatic retries. Raw-HOLD is not a crash, success, performance result, universal BLAS guarantee or 100% workload support.
