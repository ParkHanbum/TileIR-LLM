"""Offline statistics ONLY. Does not verify receipts or authorize execution."""
import math
import statistics

# Approximate two-sided 95% Student-t critical values for df1..9.
T95 = {1: 12.706204736, 2: 4.302652730, 3: 3.182446305,
       4: 2.776445105, 5: 2.570581836, 6: 2.446911851,
       7: 2.364624252, 8: 2.306004135, 9: 2.262157163}
BUDGET_RATIO = 1.05

def decide(baseline_ms, candidate_ms):
    """One positive geometric-mean wall summary per independent matched process.

    Caller must separately enforce qualification, matching and independence.
    Two to five pairs are screening only. Six to ten support the stated
    conditional approximation; no automatic extra runs or optional stopping.
    """
    baseline_ms, candidate_ms = tuple(baseline_ms), tuple(candidate_ms)
    n = len(baseline_ms)
    if n != len(candidate_ms) or not 2 <= n <= 10:
        raise ValueError('Require 2..10 matched process summaries')
    for value in (*baseline_ms, *candidate_ms):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Wall summaries must be finite positive numbers, not bool')
    ratios = [c / b for b, c in zip(baseline_ms, candidate_ms)]
    if any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError('Process ratio outside representable range')
    logs = [math.log(r) for r in ratios]
    mean = statistics.mean(logs)
    half = T95[n - 1] * statistics.stdev(logs) / math.sqrt(n)
    low, high = mean - half, mean + half
    # Compare in the SAME log-ratio domain, not expm1 vs decimal .05.
    # Identical ratios of exactly 1.05 are within budget, never regression.
    boundary = math.log(BUDGET_RATIO)
    verdict = 'INCONCLUSIVE'
    if n >= 6:
        if low > boundary and all(r > BUDGET_RATIO for r in ratios):
            verdict = 'REGRESSION'
        elif high <= boundary:
            verdict = 'WITHIN_BUDGET'
    try:
        interval = [math.exp(low), math.exp(high)]
        point = math.exp(mean)
    except OverflowError as exc:
        raise ValueError('Confidence interval outside representable range') from exc
    if not all(math.isfinite(x) and x > 0 for x in (*interval, point)):
        raise ValueError('Confidence interval outside representable range')
    return dict(verdict=verdict, independent_pairs=n, process_ratios=ratios,
                geometric_ratio=point, ratio_interval95=interval,
                budget_ratio=BUDGET_RATIO,
                method='conditional approximate paired log-ratio Student-t; process unit',
                execution_authorized=False,
                scope='fixed workload; qualification and independence checked externally')
