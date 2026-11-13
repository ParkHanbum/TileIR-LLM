# Fixed-workload v7 change qualification and performance gate

This is the integrable policy; it supersedes the exploratory PLAN.md. Test and performance-gate implementation is authorized. The remaining repeat-lifecycle issue is correctness qualification, not a permission blocker. No GPU execution or performance result is supplied by this document or the accompanying statistics module.

## Baseline and runnable identities
- Immutable provenance tag: `v7-own-kernel-fixed-workload-20261113`, commit `0b9611661ef216c35fb9f4c5da92662e5930dcbb`. Never move it or edit its numerical sources in place.
- Last parent-reported tested reproduction HEAD: `a8ffb8b`. Execution receipts must resolve and record the full commit SHA; never use an abbreviated hash as an expected-HEAD gate.
- An accepted performance baseline is a separately named runnable HEAD plus measured performance receipts. Neither the immutable provenance tag nor correctness PASS alone creates a performance baseline.
- Future intentional source changes require a reviewed, versioned source/contract manifest. Do not bypass the current baseline-byte checks. Require exact expected HEAD, clean tracked worktree, and tracked/hash-checked imported local code, including generated-code provenance. Preserve previous failures/results.

## Mandatory order
1. CPU tests for the exact proposed HEAD, including deliberate failure and corruption cases.
2. Every affected standalone GPU module: actual raw/state/readonly/alias/error checks, exact fixtures/counts and independent recovery. Missing coverage is HOLD.
3. Whole fixed workload: exact 512 token IDs and 2560 public values, strict dispatch, same-request compute census and own-source/symbol attribution; classic/Lt no-compute audit with positive control. Require actual outer/child exit0, cleanup and independent recovery. Unknown kernel attribution is not full coverage. Loaded-module SHA remains distinct from source/symbol evidence.
4. Separately qualify fresh-request same-engine repetition, including N,L,L,N, at that runnable HEAD. Require full endpoints/census/ownership/restore/retirement after every request, with permanent failure latch and no counter reset, fake observation flags, hidden native request, owner clear or engine recreation.
5. Only after all required correctness receipts match the exact HEAD and approved manifest, run SEPARATE balanced repeated warm wall. Performance never overrides correctness failure. No profiler, logger or cProfile inside the wall interval.

Stop on first failure. No automatic retries, KV adjustments, empty_cache, precision changes or additional runs selected after seeing results. Save partial receipts and perform bounded cleanup/recovery. The statistics function is NOT this admission gate: it accepts numbers only and cannot verify receipts, process independence or authorize execution.

## Original-v7 repeat contract and smallest reuse
Measured candidate arm name: **original-v7 numerical recipe / prequalified fresh-request lifecycle**. Do not label it original one-shot runtime. Do not select G/H/O/M/S or mix their performance results.

Source inspection:
- `whole_lean_extensions_runtime_v7/run.py`: original one-shot request/lifecycle.
- `native_repeat_request_v1/request.py`: source-pinned native specialization with unique request prefixes.
- `three_arm_warm_runtime_v1/request.py`: timer encloses unchanged add_request submissions, engine.step loop through completion, final device sync. Token/public extraction and validation are outside.
- `three_arm_warm_runtime_v1/native.py`: observer-free plain native binding checks outside timer, including quant_method/attention impl and absent/present instance overrides.
- `three_arm_warm_runtime_v1/sequence.py`: L is the original numerical recipe, but uses `prequalified_v7_request_v2.Cycle`; G must be removed in a two-arm specialization.
- `prequalified_v7_request_v2/{cycle,types,adapter,admission}.py`: fresh admission replaces current native-observation prerequisites. Cast/Integer numerical methods remain inherited; original Extension numerical installation/finish/close remain inherited. Zero retains fresh registry collection/check, owner identity and strong references, but removes the intervening native request. This changes state trajectory and therefore needs repeat correctness qualification. No per-input value/alias guard removal was found in those overridden bodies; this scoped reading is not a blanket dependency proof.
- `full_v7_request_factory_v1/factory.py`: original fresh native-observed pair and persistent owner lifecycle, not an observer-free warm native comparator.

Minimum integration: specialize the existing sequence to N/L only, keep its original timer AST and proven bootstrap/lifetime. Include all repeat-lifecycle sources in tracked admission. CPU tests should assert original class/method identities, only the explicit admission assertion changes in owned installation, first-failure persistence, stale callback rejection and restore-before-retire. Historical L→L proof is provenance, not a replacement for exact-HEAD repeat proof. No new numerical implementation or profiling framework is required.

## Fair measured scope
Fixed B32/P53/G16 BF16, T/T, eager mode0/graphsNONE, identical model, KV, input/seed and public API guards. One engine per process; fresh request-local contexts/counters, persistent staged/zero/UVA owners. Verify fresh source/owner/address/layout identities before requests and preserve strong references until verified sync/restore. Zero call count remains an observed/qualified workload guard, not a universal assumption. Record per-request peak allocation/reservation and retained/retired reference counts; two-request stability does not establish absence of long-run leakage.

Plain N has no reference observer/hash/copy in its timed path. L retains its existing public per-operation guards. Installation, admission hashing, descriptor preparation, output proof, finish/close and owner retirement are outside the timer. Pre-sync is outside; final complete sync is inside. This is **prepared-request wall**, not full service latency or installation cost. Untimed N/L warmups must pass before timing; compilation is warmed at supported shapes, but lack of compile telemetry is not proof that no later compilation occurred.

Minimal screening schedules in independent processes: N,L,L,N and L,N,N,L, after full untimed warmups. Preserve raw timings and order. Across implementation comparisons, match process pairs by predeclared index/environment and alternate baseline-first versus candidate-first, not by nearest observed duration. Fresh external roots/cache/temp/dumps, no overwrite, common lock, bounded child/parent timeouts and independent recovery follow regression_v7. Preserve GUI processes; do not kill the desktop or claim exclusive device access. Coordinator admission should record competing activity/thermal/power/clock conditions outside timing; disturbances are HOLD or INCONCLUSIVE, not silently discarded samples.

## Performance decision
Compare candidate implementation wall to a named accepted runnable performance baseline, not to native speed. Report native and candidate absolute wall and their normalized ratio separately; normalization must not hide an absolute slowdown. Each independent process contributes one geometric mean of its timed implementation requests. Requests within that process are correlated and are not independent samples.

Predeclare the number of matched independent process pairs. Two to five pairs are screening only and always INCONCLUSIVE in the supplied function. Six to ten pairs permit a conditional approximate 95% paired log-ratio Student-t interval. Six is an operational minimum, not a power or distribution guarantee. For greater counts or nonstationarity, review a separate analysis plan rather than silently extending the bounded function. A minimal identical-arm control can assess noise before accepting the model; none is run here.

Let ratio = candidate/reference and budget = 1.05:
- REGRESSION: lower conditional 95% ratio bound >1.05 and every process-pair point ratio >1.05.
- WITHIN_BUDGET: upper bound <=1.05.
- Otherwise INCONCLUSIVE, including noisy nonsignificance.

Five percent is a chosen engineering budget, not an empirical universal noise bound. Report all process ratios, number of independent pairs, point estimate, interval, assumptions, raw wall values, GUI conditions and fixed-workload scope. Identical ratios exactly1.05 are within budget. Comparisons occur in log-ratio space to avoid expm1 rounding at the 5% boundary. No tolerance band or optional stopping is hidden in the decision. Statistical status alone never grants permission or establishes correctness.

## Integrable files and CPU-only test
`performance_decision.py` is stdlib-only, with no tool/framework/GPU imports or side effects. `test_performance_decision.py` supports package-relative or standalone execution. Run with CUDA_VISIBLE_DEVICES='' and PYTHONDONTWRITEBYTECODE=1. Tests cover exact5%, adjacent representable ratios, known ratios, noisy/direction-disagreeing pairs, n<6 screening, invalid bool/NaN/infinity/nonpositive values, unequal counts and unrepresentable arithmetic.
