# YYYY-MM-DD — Experiment title

Every section below is required. Use “not measured”, “not applicable” or an
explicit missing prerequisite rather than leaving evidence gaps implicit.
Write for an engine developer deciding whether and how to apply the change.

## Track, question and hypothesis
Identify BLAS kernel qualification, runtime-flow work or a matched end-to-end
checkpoint. Separate established facts from hypotheses and planned work.

## Where the change lives
Show the relevant export / dispatch / callback / request-lifecycle flow.
Identify the changed boundary, its caller and consumer, ownership and completion
requirements. Distinguish standalone qualification from integrated deployment.

## Exact before → after
Provide concrete snippets or a per-field table, source/artifact identities and
actual changed facts. Label pseudocode as pseudocode. Do not write merely
“some settings changed”. List held constants (source, ABI, numeric contract,
tile/hints, input fixtures and checks as applicable). Separate bundled changes
from one-factor tests; do not attribute a bundle to one fact without evidence.

## Preconditions and checked dispatch contract
Specify dimensions, strides in elements versus bytes, dtype/accumulation/store
rounding, layout, ABI and supported shape ranges. Check actual view pointers
(owner payload plus offset), not only owner alignment. State lifetime,
completion and alias requirements when relevant; do not invent assumptions.

## Environment and workload
GPU/driver/toolchain, model revision, fixture hashes and provenance, shapes,
precision, sampling/EOS, runtime configuration and effective dispatch mode.
Distinguish true per-layer activations from reused or synthetic inputs.

## Validation and failure behavior
Describe positive cases, tested boundaries/tails, negative cases and expected
rejection/fallback. State what actually ran, exactness/tolerance criteria,
compared element/token counts, actual exits and independent resource recovery.
Separate correctness-only paths from timed paths and mark untested cases.

## Generated-code evidence
Record verified compiler/export/ELF/IR/PTX/SASS differences and their scope.
Separate static register/shared/stack/instruction facts from measured dynamic
utilization, memory traffic and stalls. Do not claim a pass, lowering path,
instruction or causal mechanism that has not been observed.

## Measurement boundary
Show preparation → warmup → operation/request → output readiness → cleanup.
Mark synchronization, checks, logging and compilation inside or outside timing.
Distinguish event envelopes, profiler kernel duration and host wall time;
document nested/cross-stream overlap, submission gaps and cache perturbations.

## Results and measured significance
Provide arm identities, chronological order, sample counts, medians and spread;
link sanitized public evidence when available. Label rounded samples and
summary-only datasets. Explain effect size and drift/confounding honestly.
Do not chain separate experiments or mix stale reference times into a new ratio.

## Applicability, risk and fallback
State exactly which caller/workloads can use the change, which cannot, and
what validation is required before dispatch. Explain failure risks if assumptions
are violated and the actual available fallback; label a proposed fallback as
unimplemented. Do not turn one fixture into a general performance guarantee.

## Reproduction and missing prerequisites
Include commands only when actually verified and suitable for public use.
Supply accessible source/config/fixture identities and exact contracts. If
commands or artifacts are unavailable, list the missing prerequisites instead
of inventing commands or claiming full reproducibility. Exclude private paths,
credentials and non-public raw inputs.

## Interpretation, next discriminating test and corrections
State what the experiment proves, leaves unresolved and cannot establish.
Describe the next controlled test without presenting it as completed.
Date and explain later corrections while preserving historical measured scope.
