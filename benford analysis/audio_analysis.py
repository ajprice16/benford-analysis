import librosa
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

def get_n_fft_from_tempo(audio_path, beat_fraction=1.0):
    """Calculate optimal FFT window size based on audio tempo."""
    y, sr = librosa.load(audio_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo)
    beat_duration_sec = 60.0 / tempo
    window_duration_sec = beat_duration_sec * beat_fraction
    n_fft = int(window_duration_sec * sr)
    return 2 ** int(np.round(np.log2(n_fft)))

def extract_dominant_frequencies(audio_path, n_fft=2048, top_n=100):
    """Extract dominant frequencies from audio file."""
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    S = np.abs(librosa.stft(y, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    dominant_freqs = []
    for i in range(S.shape[1]):
        peak_idx = np.argmax(S[:, i])
        if peak_idx < len(freqs):
            freq = freqs[peak_idx]
            if 20 <= freq <= 20000:
                dominant_freqs.append(freq)
    
    freq_counts = Counter(np.round(dominant_freqs))
    return [freq for freq, _ in freq_counts.most_common(top_n)]

def analyze_audio_file(audio_path, beat_fraction=1.0, top_n=100):
    """Full analysis pipeline for audio files."""
    n_fft = get_n_fft_from_tempo(audio_path, beat_fraction)
    freqs = extract_dominant_frequencies(audio_path, n_fft, top_n)
    print("Success")
    return pd.DataFrame(freqs)