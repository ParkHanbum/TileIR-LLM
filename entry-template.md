# YYYY-MM-DD — Experiment title

## Question and hypothesis
State what is known and what is only inferred.

## Exact A/B difference
Source IDs, changed files, retained numerical path and safety contracts.

## Environment and workload
GPU/driver/toolchain, model revision, input fixture hashes, shapes, precision,
sampling/EOS, runtime config and effective dispatch mode.

## Measurement boundary
Preparation, warmup, request reset, inference, output readiness, cleanup,
synchronization, comparison and logging. State which are timed.

## Validation before timing
Positive/negative tests, expected outputs, actual exits, failure handling.

## Raw results and summary
Link sanitized raw data; block order and repetitions; medians and spread.
Do not combine event intervals, profiler kernel durations and wall time.

## Correctness and recovery
Token counts, reference, intermediate scope, exits and resource recovery.

## Interpretation and limits
What the experiment proves, does not prove, and next discriminating test.

## Reproduction
Actual tested commands and accessible source/config/input manifests.
If unavailable, say so rather than claiming full reproducibility.

## Corrections
Date and reason for any later correction.
