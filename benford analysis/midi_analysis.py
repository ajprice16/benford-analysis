import mido
import pandas as pd

def note_to_freq(note):
    """Convert MIDI note number to frequency."""
    return 440 * (2**((note - 69) / 12))

def parse_midi(file_path):
    """Extract note frequencies from MIDI file."""
    notes = []
    try:
        midi_file = mido.MidiFile(file_path)
        for i, track in enumerate(midi_file.tracks):
            for msg in track:
                if not msg.is_meta and msg.type == "note_on":
                    notes.append(msg.note)
        
        if not notes:
            raise ValueError(f"Empty MIDI file: {file_path}")
        
        return [round(note_to_freq(note)) for note in notes]
    except Exception as e:
        raise ValueError(f"Error processing MIDI file {file_path}: {str(e)}")

def analyze_midi_file(midi_path):
    """Full analysis pipeline for MIDI files."""
    frequencies = parse_midi(midi_path)
    return pd.DataFrame(frequencies)