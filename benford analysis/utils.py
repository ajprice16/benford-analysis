import numpy as np
from scipy import stats

def get_props(observed, expected):
    """Calculate proportions from observed and expected counts."""
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    return observed / observed.sum(), expected / expected.sum()

def delta(observed, expected):
    """Sum of absolute differences between proportions."""
    obs_p, exp_p = get_props(observed, expected)
    return np.sum(np.abs(obs_p - exp_p))

def NED(observed, expected):
    """Normalized Euclidean Distance."""
    obs_p, exp_p = get_props(observed, expected)
    return np.sqrt(np.sum(np.square(obs_p - exp_p) / exp_p))

def MAD(observed, expected, benford_probs):
    """Mean Absolute Deviation."""
    obs_p = np.asarray(observed, dtype=float) / np.sum(observed)
    return np.abs(obs_p - benford_probs).mean()

def zStat(observed, expected):
    """Maximum standardized difference."""
    obs_p, exp_p = get_props(observed, expected)
    std = np.sqrt(exp_p * (1 - exp_p) / np.sum(observed))
    return np.max(np.abs(obs_p - exp_p) / std)

def calculate_metrics(observed_counts, expected_counts, benford_probs):
    """Calculate all metrics for Benford analysis."""
    return {
        'delta': delta(observed_counts, expected_counts),
        'NED': NED(observed_counts, expected_counts),
        'MAD': MAD(observed_counts, expected_counts, benford_probs),
        'zStat': zStat(observed_counts, expected_counts),
        'pearson': stats.pearsonr(
            observed_counts / np.sum(observed_counts),
            expected_counts / np.sum(expected_counts)
        )[0]
    }