from benford_analysis import (
    analyze_audio_file,
    analyze_midi_file,
    benford_combined_test,
    plot_benford_distribution,
    calculate_metrics
)
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def run_analysis(file_path, is_midi=False):
    """Run full analysis on a single file."""
    try:
        print(f"\nProcessing {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None

        # Load and parse file
        if is_midi:
            print("Loading as MIDI...")
            data = analyze_midi_file(file_path)
        else:
            print("Loading as audio...")
            data = analyze_audio_file(file_path)
        
        if data is None:
            print("No data returned from analysis")
            return None
        if len(data) == 0:
            print("Empty dataset returned")
            return None
            
        print(f"Successfully loaded {len(data)} data points")
        
        # Benford analysis
        benford_probs = [np.log10(1 + 1/d) for d in range(1, 10)]
        digit_counts = data[0].astype(str).str[0].value_counts().sort_index()
        observed_counts = [digit_counts.get(str(d), 0) for d in range(1, 10)]
        expected_counts = [p * len(data) for p in benford_probs]
        
        # Calculate metrics
        metrics = calculate_metrics(observed_counts, expected_counts, benford_probs)
        benford_test = benford_combined_test(data)
        
        print("Analysis completed successfully")
        
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
        print(f"\nERROR processing {file_path}:")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting Benford analysis test...")
    
    # Set working directory
    base_dir = "/Users/ajpri/Summer"
    os.chdir(base_dir)
    print(f"Working directory: {os.getcwd()}")
    
    # Verify directories
    wav_dir = "./Wavs"
    midi_dir = "./Midis"
    
    if not os.path.exists(wav_dir):
        print(f"Error: Directory not found - {wav_dir}")
        exit(1)
        
    if not os.path.exists(midi_dir):
        print(f"Error: Directory not found - {midi_dir}")
        exit(1)
    
    # Find files
    audio_files = [f for f in os.listdir(wav_dir) if f.lower().endswith(('.wav', '.mp3'))]
    midi_files = [f for f in os.listdir(midi_dir) if f.lower().endswith('.mid')]
    
    print(f"\nFound {len(audio_files)} audio files")
    print(f"Found {len(midi_files)} MIDI files")
    
    if not audio_files and not midi_files:
        print("No files found to process")
        exit(0)
    
    # Process files
    results = []
    counts_totals = []
    filenames = []
    
    for file in audio_files:
        full_path = os.path.join(wav_dir, file)
        result = run_analysis(full_path)
        if result:
            results.append(result)
            counts_totals.append(
                np.array(result['observed_counts']) / sum(result['observed_counts'])
            )
            filenames.append(file)
    
    for file in midi_files:
        full_path = os.path.join(midi_dir, file)
        result = run_analysis(full_path, is_midi=True)
        if result:
            results.append(result)
            counts_totals.append(
                np.array(result['observed_counts']) / sum(result['observed_counts'])
            )
            filenames.append(file)
    
    # Save and plot results
    if results:
        print("\nSaving results...")
        pd.DataFrame(results).to_csv('benford_results.csv')
        print("Saved results to benford_results.csv")
        
        print("Generating plot...")
        plot_benford_distribution(counts_totals, filenames, 'benford_plot.png')
        plt.show()  # Ensure plot displays
        print("Plot saved to benford_plot.png")
    else:
        print("\nNo valid results to save")
    
    print("\nAnalysis complete!")