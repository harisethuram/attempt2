generatemidi.py
from midiutil import MIDIFile
# from preprocess import preprocess
import torch

def generate_midi(midis: 'list[torch.LongTensor[D, 4]]', name):
    """
    - takes in notes (d,4) tensor where each row is a note, col 0 is pitch, col 1 is division, col 2 is beat, col 3 is tatum
    - generates a midi file for the notes
    """
    for i, midi in enumerate(midis):
        process_midi(midi, f'{name}{i}')

def process_midi(notes, filename):
    print(notes.shape)
    track = 0
    channel = 0
    time = 0
    tempo = 220
    volume = 100
    midi = MIDIFile(1)
    midi.addTempo(track, time, tempo)
    for i in range(1, notes.shape[0]):
        if notes[i, 1] == 0 and notes[i, 0] == 0:
            break
        pitch = notes[i, 0]
        duration = notes[i, 1]
        if pitch != 0:
            midi.addNote(track, channel, int(notes[i, 0]), time, notes[i, 1], volume)
        time += duration

    with open(f"{filename}.mid", "wb") as output_file:
        midi.writeFile(output_file)
