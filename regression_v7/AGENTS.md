# Regression changes

Follow WORKFLOW.md in this directory. CPU -> affected standalone GPU modules ->
whole exact/census/no-cuBLAS -> separately qualified repeated warm wall. Record
full immutable tested HEAD and actual exits/recovery. Stop on failure.
Do not move the baseline tag or mix optimized candidates into original v7.
Only the coordinator writes Git or schedules GPU work. Local only, no push.
No model, cache, tensor dumps, binaries, credentials or raw logs in Git.
Report dedicated module scope separately from E2E-only scope; no universal input
claim. A 5% regression budget is engineering policy, not measured noise.
