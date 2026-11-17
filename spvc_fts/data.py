"""Fuzzification formulas from section 3 of Chen and Lu (2026)."""
import numpy as np
from scipy.special import ndtr, ndtri


def financial_returns(low, high):
    """Convert price intervals to (centre, left spread, right spread).

    The paper's cross-ratio log return is used as printed, in natural-log
    units, with no unspecified multiplication by 100.
    """
    low, high = np.asarray(low, float), np.asarray(high, float)
    if low.ndim != 1 or low.shape != high.shape or len(low) < 2:
        raise ValueError('low and high must be equal-length 1-D arrays')
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(low <= 0) or np.any(high < low):
        raise ValueError('prices must be finite and satisfy 0 < low <= high')
    a = np.log(low[1:] / high[:-1])
    b = np.log(high[1:] / low[:-1])
    return np.column_stack(((a+b)/2, (b-a)/2, (b-a)/2))


def temperature_fuzzy(values, probability=0.05):
    """Normal-probability fuzzification, population standard deviation.

    Requires each tail to contain at least the requested probability; the
    paper does not say how to handle observations violating this condition.
    """
    x = np.asarray(values, float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError('values must be a finite 1-D series')
    mu, sigma = x.mean(), x.std(ddof=0)
    if sigma == 0 or not 0 < probability < .5:
        raise ValueError('nonzero standard deviation and 0 < probability < 0.5 required')
    p = ndtr((x-mu)/sigma)
    if np.any((p <= probability) | (p >= 1-probability)):
        raise ValueError('normal tail probability is insufficient for a finite spread')
    lower = mu + sigma*ndtri(p-probability)
    upper = mu + sigma*ndtri(p+probability)
    return np.column_stack((x, x-lower, upper-x))
