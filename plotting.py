import matplotlib.pyplot as plt
import numpy as np

def plot_benford_distribution(counts_totals, filenames, save_path=None):
    """Plot observed distributions against Benford's Law."""
    digits = np.arange(1, 10)
    benford_probs = [np.log10(1 + 1/d) for d in digits]
    
    plt.figure(figsize=(10, 6))
    plt.plot(digits, benford_probs, 'k-', linewidth=2, label="Benford's Law")
    
    for i, counts in enumerate(counts_totals):
        plt.plot(digits, counts, 'o-', label=filenames[i] if i < len(filenames) else f"File {i+1}")
    
    plt.title("First Digit Distribution vs Benford's Law")
    plt.xlabel("First Digit")
    plt.ylabel("Proportion")
    plt.xticks(digits)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()