from benford_analysis import (
    analyze_audio_file,
    analyze_midi_file,
    benford_combined_test,
    plot_benford_distribution,
    calculate_metrics
)
import pandas as pd
import os

def run_analysis(file_path, is_midi=False):
    """Run full analysis on a single file."""
    try:
        # Load and parse file
        if is_midi:
            data = analyze_midi_file(file_path)
        else:
            data = analyze_audio_file(file_path)
        
        # Benford analysis
        benford_probs = [np.log10(1 + 1/d) for d in range(1, 10)]
        digit_counts = data[0].astype(str).str[0].value_counts().sort_index()
        observed_counts = [digit_counts.get(str(d), 0) for d in range(1, 10)]
        expected_counts = [p * len(data) for p in benford_probs]
        
        # Calculate metrics
        metrics = calculate_metrics(observed_counts, expected_counts, benford_probs)
        benford_test = benford_combined_test(data)
        
        return {
            'filename': os.path.basename(file_path),
            'type': 'MIDI' if is_midi else 'Audio',
            **metrics,
            'fisher_stat': benford_test['combined_stat'],
            'p_value': benford_test['combined_p'],
            'is_benford': benford_test['combined_p'] > 0.05,
            'observed_counts': observed_counts
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    audio_files = [f for f in os.listdir('audio_dir') if f.endswith(('.wav', '.mp3'))]
    midi_files = [f for f in os.listdir('midi_dir') if f.endswith('.mid')]
    
    results = []
    counts_totals = []
    filenames = []
    
    for file in audio_files:
        result = run_analysis(os.path.join('audio_dir', file))
        if result:
            results.append(result)
            counts_totals.append(
                np.array(result['observed_counts']) / sum(result['observed_counts'])
            )
            filenames.append(file)
    
    for file in midi_files:
        result = run_analysis(os.path.join('midi_dir', file), is_midi=True)
        if result:
            results.append(result)
            counts_totals.append(
                np.array(result['observed_counts']) / sum(result['observed_counts'])
            )
            filenames.append(file)
    
    # Save and plot results
    pd.DataFrame(results).to_csv('benford_results.csv')
    plot_benford_distribution(counts_totals, filenames, 'benford_plot.png')