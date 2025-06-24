import numpy as np
from scipy import stats
from scipy.stats import chi2, uniform
from tqdm import tqdm

def benford_prob(d):
    """Probability of digit d according to Benford's Law."""
    return np.log10(1 + 1/d)

def get_first_digit(x):
    """Extract first significant digit from array of numbers."""
    x = np.asarray(x)
    mask = (x != 0) & ~np.isnan(x)
    x = np.abs(x[mask])
    return np.floor(x / 10**np.floor(np.log10(x))).astype(int)

def get_significand(x):
    """Extract significand (mantissa) from array of numbers."""
    x = np.asarray(x)
    mask = (x != 0) & ~np.isnan(x)
    x = np.abs(x[mask])
    return 10**(np.log10(x) % 1)

def z_transform(x, d):
    """Z-transform for Hotelling test."""
    s = get_significand(x)
    return (10 * s) * ((10 * s) >= d) * ((10 * s) < d + 1)

def benford_combined_test(data, B=1000, show_progress=False):
    """Combined Pearson χ² and Hotelling test for Benford's Law."""
    DIGITS = np.arange(1, 10)
    C = np.log10(np.e)
    data = np.asarray(data)
    n = len(data)
    
    # Pearson χ² test
    expected_counts = np.array([benford_prob(d) * n for d in DIGITS])
    observed_counts = np.array([np.sum(get_first_digit(data) == d) for d in DIGITS])
    chi2_stat = np.sum((observed_counts - expected_counts)**2 / expected_counts)
    chi2_p = max(1 - chi2.cdf(chi2_stat, df=8), 1e-10)
    
    # Hotelling Q test
    z_bars = np.array([np.mean(z_transform(data, d)) for d in DIGITS])
    var_z = np.array([C * (d + 0.5 - C) for d in DIGITS])
    cov_z = -C**2
    sigma = np.diag(var_z) + cov_z * (1 - np.eye(9))
    diff = z_bars - C
    q_stat = n * diff @ np.linalg.inv(sigma) @ diff
    
    # Monte Carlo simulation
    def generate_benford_sample(n):
        return 10**uniform.rvs(size=n)
    
    q_samples = []
    iter_range = tqdm(range(B), desc="Simulating") if show_progress else range(B)
    for _ in iter_range:
        sample = generate_benford_sample(n)
        test_z = np.array([np.mean(z_transform(sample, d)) for d in DIGITS])
        q_samples.append(n * (test_z - C) @ np.linalg.inv(sigma) @ (test_z - C))
    
    hotelling_p = (np.sum(np.array(q_samples) >= q_stat) + 1) / (B + 1)
    
    # Fisher's combined test
    fisher_stat = -2 * (np.log(chi2_p) + np.log(hotelling_p))
    combined_p = max(1 - chi2.cdf(fisher_stat, df=4), 1e-10)
    
    return {
        'combined_stat': fisher_stat,
        'combined_p': combined_p,
        'chi2_stat': chi2_stat,
        'chi2_p': chi2_p,
        'hotelling_q': q_stat,
        'hotelling_p': hotelling_p,
        'digits': {d: int(observed_counts[i]) for i, d in enumerate(DIGITS)},
        'expected': {d: float(expected_counts[i]) for i, d in enumerate(DIGITS)}
    }