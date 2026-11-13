# CPU preparation: ordinary decode recurrent core only
Eager native engine True/True read-only, compile0/graphsNONE, no performance. Public imported packed-decode helper only;24owners×15=360 observed calls asserted. Native prefill/convolution/postconv remain untouched and binding identities checked; their call counts are not independentlyobserved here.

Shadow exactlyone native helper call returns original tuple, actual full selected32state slots compared against candidate on separate compact[33,32,128,128]FP32 scratch. Positive unique32indices remapped1..32; actual metadata retained. Actual source state V,K orientation retained but pagepitch/offset remapped inshadow. Slot0sentinel protected. No realstate restore/writes bycandidate,no full5GiBowner or670MBminspan clones/hashes. Two selected64MiB snapshots plus66MiB scratch; metrics temporarybuffers GPU only, small scalars CPU. First mismatch saves readonlyinputs+pre/expected/candidate selectedstate+output, then immediately stops the request after completed native call; replaceblockedunlessall360pass. Original isolatedbank verifies actualpitch/offset anduntouched minspan; cannot extrapolate that into wholeowner verification at everymodelcall.

Replacement usesactualstate/X/out/indices metadata, same-storage detaches for Parameter inputs,returnsoriginal callerout/state objects.360candidate calls and native0 calls, fallback0. Same-run512IDs/public2560 exact gates, unchanged cleanup/admission900s, no adoption beforeactualexit/recovery. Cachepreseed parent-owned.

## Frozen token companion
Existing nativecapture EXPECTED-TOKENS.json SHA1c5b9bd9fbcbd7bb14691a3da35698877a06a2ef6da7b79c1239ed5bdf6598b9 is pinned read-only, loaded/shapechecked before CPUdry return, passed explicitly by launcher and defaulted in directrunner. Both shadow/replacement must match all512 frozenIDs in addition to same-run IDs/public2560 exact. Parent fixed-gate request; previousv1/v2 source preserved. No numericalAPIchanges.

## Runtimev4 narrow extension
Relative to runtimev3 only candidate APIv4 import, qualificationfunction/receipt and module namespace change. New2bank regression binds originalnative,oldAPIrepro,andnewnativeequality. Previousfullruntimev3 first7shadow6PASS+layer8outputHOLD preserved. FixedEXPECTED512,same-run public2560 exact,rawstate/output firstfailurestop,allfinite/cleanup unchanged. No runtimev4 GPUyet.
