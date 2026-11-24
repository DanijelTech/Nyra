"""Statistics used by the critic and the safety gate.

Everything is implemented here, in the standard library, for three reasons: the
gate must be reproducible on any machine (no scipy/numpy in the critical path),
the numbers must be auditable line by line, and a statistical shortcut that we
cannot explain is not allowed to block a deployment.

The tests pin closed-form values (Wilson intervals, z-tests) against published
examples so a regression in this file is caught immediately.
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def _clean(values: Sequence[float]) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def mean(values: Sequence[float]) -> float:
    data = _clean(values)
    return sum(data) / len(data) if data else 0.0


def median(values: Sequence[float]) -> float:
    data = sorted(_clean(values))
    if not data:
        return 0.0
    mid = len(data) // 2
    if len(data) % 2:
        return data[mid]
    return 0.5 * (data[mid - 1] + data[mid])


def mad(values: Sequence[float]) -> float:
    """Median absolute deviation (unscaled)."""
    data = _clean(values)
    if not data:
        return 0.0
    centre = median(data)
    return median([abs(v - centre) for v in data])


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    data = sorted(_clean(values))
    if not data:
        return 0.0
    if q <= 0:
        return data[0]
    if q >= 100:
        return data[-1]
    pos = (len(data) - 1) * (q / 100.0)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return data[low]
    return data[low] + (data[high] - data[low]) * (pos - low)


def relative_delta(baseline: float, current: float) -> float | None:
    """Signed relative change. ``None`` when the baseline is zero (undefined)."""
    if baseline == 0:
        return None
    return (current - baseline) / abs(baseline)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_sided_p_from_z(z: float) -> float:
    return max(0.0, min(1.0, 2.0 * (1.0 - normal_cdf(abs(z)))))


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a proportion. Deterministic, no continuity correction."""
    if n <= 0:
        return (0.0, 1.0)
    successes = max(0, min(n, successes))
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def two_proportion_z(s1: int, n1: int, s2: int, n2: int) -> tuple[float, float]:
    """Two-sided z-test for equal proportions. Returns ``(z, p)``; ``(0, 1)`` if degenerate."""
    if n1 <= 0 or n2 <= 0:
        return (0.0, 1.0)
    p1, p2 = s1 / n1, s2 / n2
    pooled = (s1 + s2) / (n1 + n2)
    variance = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    if variance <= 0:
        return (0.0, 1.0)
    z = (p1 - p2) / math.sqrt(variance)
    return (z, two_sided_p_from_z(z))


def bootstrap_median_ci(
    values: Sequence[float],
    *,
    iterations: int = 2000,
    alpha: float = 0.05,
    seed: int = 20240501,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the median. Seeded, therefore reproducible."""
    data = _clean(values)
    if not data:
        return (0.0, 0.0)
    if len(data) == 1:
        return (data[0], data[0])
    rng = random.Random(seed)
    n = len(data)
    medians = []
    for _ in range(iterations):
        sample = [data[rng.randrange(n)] for _ in range(n)]
        medians.append(median(sample))
    medians.sort()
    lo_idx = int(math.floor((alpha / 2) * len(medians)))
    hi_idx = int(math.ceil((1 - alpha / 2) * len(medians))) - 1
    return (medians[max(0, lo_idx)], medians[min(len(medians) - 1, hi_idx)])


def stdev(values: Sequence[float]) -> float:
    data = _clean(values)
    if len(data) < 2:
        return 0.0
    centre = mean(data)
    return math.sqrt(sum((v - centre) ** 2 for v in data) / (len(data) - 1))


def changepoint_index(
    values: Sequence[float],
    *,
    min_segment: int = 8,
    threshold: float = 4.0,
) -> int | None:
    """Locate a sustained level shift by binary segmentation.

    For every split that leaves at least ``min_segment`` samples on both sides we
    form the Welch z-statistic of the two means and keep the strongest. The shift
    is reported only if that z reaches ``threshold`` (default 4σ), which keeps
    ordinary noise from inventing a changepoint while still catching the kind of
    step a policy regression produces.

    Returns the index of the first sample *after* the shift, or ``None``.
    """
    data = _clean(values)
    if len(data) < 2 * min_segment:
        return None
    best_split: int | None = None
    best_z = 0.0
    for split in range(min_segment, len(data) - min_segment + 1):
        left, right = data[:split], data[split:]
        left_mean, right_mean = mean(left), mean(right)
        left_var = stdev(left) ** 2 / len(left)
        right_var = stdev(right) ** 2 / len(right)
        denominator = math.sqrt(left_var + right_var)
        if denominator <= 0:
            # Zero-variance segments: a difference is a genuine step, not noise.
            if left_mean != right_mean:
                return split
            continue
        z = abs(right_mean - left_mean) / denominator
        if z > best_z:
            best_z, best_split = z, split
    if best_split is not None and best_z >= threshold:
        return best_split
    return None


#: Backwards-compatible alias.
cusum_changepoint = changepoint_index


def change_ratio(baseline: Sequence[float], current: Sequence[float]) -> float | None:
    """Median ratio current/baseline, ``None`` when undefined."""
    b = median(baseline)
    c = median(current)
    if b == 0:
        return None
    return c / b


def summarise(values: Sequence[float]) -> dict[str, float]:
    data = _clean(values)
    if not data:
        return {"n": 0, "median": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0, "mad": 0.0}
    return {
        "n": float(len(data)),
        "median": median(data),
        "mean": mean(data),
        "p95": percentile(data, 95),
        "max": max(data),
        "mad": mad(data),
    }
