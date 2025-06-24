import mido
import numpy as np
import pandas as pd
from collections import defaultdict

def note_to_freq(note):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((note - 69) / 12.0))

def parse_midi(file_path):
    """
    Extract note frequencies from MIDI file with error handling.
    Returns a list of frequencies in Hz.
    """
    try:
        midi_file = mido.MidiFile(file_path)
    except Exception as e:
        raise ValueError(f"Failed to load MIDI file {file_path}: {str(e)}")

    notes = []
    
    for msg in midi_file:
        # Skip meta messages and non-note messages
        if msg.is_meta or msg.type not in ['note_on', 'note_off']:
            continue
            
        # Handle note messages
        note = msg.note
        if msg.type == 'note_on' and msg.velocity > 0:
            notes.append(note_to_freq(note))
    
    if not notes:
        raise ValueError(f"No valid notes found in {file_path}")
    
    return notes

def analyze_midi_file(midi_path):
    """Full analysis pipeline for MIDI files with proper error handling."""
    try:
        frequencies = parse_midi(midi_path)
        if not frequencies:
            print(f"Warning: No frequencies extracted from {midi_path}")
            return None
        
        # Create DataFrame with explicit column name
        return pd.DataFrame(frequencies, columns=['frequency'])
    
    except Exception as e:
        print(f"Error processing MIDI file {midi_path}: {str(e)}")
        return None