# Seven-adapter composition preparation only: RELEASE BLOCKED
Skip six-adapter model execution. Extend qualified v2 composition only after samplecounts full16 integration proof/source is final. No run.py/launch.py or GPUrelease artifact created at this stage. Other worker files untouched.

Planned components/counts: norm1040, attention_gate128, block_gather16, slot_mapping16, pos_seq16, gated_norm384, sample_counts16 =1616logical calls per request. These are not physical GPU launches, GPUtime share or100%coverage.

Keep existing first5 order and append gated_norm then sample_counts, unless samplecounts source reveals a shared binding requiring a documented order. Restore exact reverse; failure during installation unwinds completed components and failing component's own cleanup contract. Existing gather/slot prepare_attn nesting stays unchanged. Gated24RMSNormGated instances do not overlap65GemmaRMSNorm instances. Samplecounts binding/wrapper interaction remains pending its actual source, not assumed safe.

Gated standalone integration v2/gatedIntegration02 passed defaultT/T384shadow+384replacement,213909504raw elements,512IDs/public2560exact. RESULTafbdd75c2994d1b1062fb5758655a935ad422af5f83263c1518c0f32b905fc75, recovery2d7eff315a567e4aadaa94f82e534822df6757fe3511836becb7a13f8affb072. It does NOT prove behavior under compositionF(false,true). Seven-adapter composition must run all1616rawshadow calls under sharedF before replacement. Any newgatedinputdiff holds whole replacement; no tolerance changes.

Release prerequisites: samplecounts actual full16 shadow/replacement, originalhelper0 onreplacement,512IDs/publiclogprob gate,actualexit0 andindependentrecovery; final immutable sourcepins; all7 installation/nestedbinding/restore tests; source entryimport smoke; explicit subgates/counts/fallbacks and currentGPUwindow approval. Representative2case samplerqualification is insufficient. Neither receiptabsence nor in-progress worker status authorizes release.

No speed/optimizedfusion/fullvocab claims; combined512IDs+returnedpublic2560 only. Existing input-specific limits includingseqpadding0 remain. CPU skeleton below checks declaredcount and proofcompleteness only, not semanticqualification.
