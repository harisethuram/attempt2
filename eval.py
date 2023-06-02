import torch
import numpy as np

def inside_outside_metric_rc(notes):
    """
    - harmony: (l, 1) 
    - notes: (l2, 2)
    - returns inside_outside_metric for rhythm changes in Bb
    """
    harmony = np.array(["Bbj7", "Bbj7", "G7", "G7", "C-7", "C-7", "F7", "F7", # A1
                        "Bbj7", "Bbj7", "G7", "G7", "C-7", "C-7", "F7", "F7", 
                        "F-7", "F-7", "Bb7", "Bb7", "Ebj7", "Ebj7", "Ab7", "Ab7", 
                        "D-7", "D-7", "G7", "G7", "C-7", "C-7", "F7", "F7",
                        "Bbj7", "Bbj7", "G7", "G7", "C-7", "C-7", "F7", "F7", # A2
                        "Bbj7", "Bbj7", "G7", "G7", "C-7", "C-7", "F7", "F7", 
                        "F-7", "F-7", "Bb7", "Bb7", "Ebj7", "Ebj7", "Ab7", "Ab7", 
                        "D-7", "D-7", "G7", "G7", "C-7", "C-7", "F7", "F7",
                        "D7", "D7", "D7", "D7", "D7", "D7", "D7", "D7", # B
                        "G7", "G7", "G7", "G7", "G7", "G7", "G7", "G7", 
                        "C7", "C7", "C7", "C7", "C7", "C7", "C7", "C7", 
                        "F7", "F7", "F7", "F7", "F7", "F7", "F7", "F7",
                        "Bbj7", "Bbj7", "G7", "G7", "C-7", "C-7", "F7", "F7", # A3
                        "Bbj7", "Bbj7", "G7", "G7", "C-7", "C-7", "F7", "F7", 
                        "F-7", "F-7", "Bb7", "Bb7", "Ebj7", "Ebj7", "Ab7", "Ab7", 
                        "C-7", "C-7", "F7", "F7", "Bbj7", "Bbj7", "Bbj7", "Bbj7"
                        ])

    onsets = np.floor(np.roll(np.cumsum(notes[:, 1]), shift=-1)) # beats in which notes begin
    print(onsets[:20])
    chord_profiles = {
        "Bbj7": [38, 41, 45, 46, 50, 53, 57, 58, 62, 65, 69, 70, 74, 77, 81, 82, 86, 89, 93, 94],
        "G7": [38, 50, 62, 74, 86, 41, 53, 65, 77, 89, 43, 55, 67, 79, 91, 47, 59, 71, 83, 95], 
        "C-7": [36, 48, 60, 72, 84, 96, 39, 51, 63, 75, 87, 43, 55, 67, 79, 91, 46, 58, 70, 82, 94], #CEbGBb
        "F7": [41, 53, 65, 77, 89, 45, 57, 69, 81, 93, 39, 51, 63, 75, 87, 36, 48, 60, 72, 84, 96], # FAEbC
        "F-7": [41, 53, 65, 77, 89, 44, 56, 68, 80, 92, 39, 51, 63, 75, 87, 36, 48, 60, 72, 84, 96],
        "Bb7": [38, 41, 44, 46, 50, 53, 56, 58, 62, 65, 68, 70, 74, 77, 80, 82, 86, 89, 92, 94],
        "Ebj7": [38, 50, 62, 74, 86, 39, 51, 63, 75, 87, 43, 55, 67, 79, 91, 46, 58, 70, 82, 94], #DEbGBb
        "Ab7": [44, 56, 68, 80, 92, 39, 51, 63, 75, 87, 36, 48, 60, 72, 84, 96, 42, 54, 66, 78, 90],
        "D-7": [41, 53, 65, 77, 89, 45, 57, 69, 81, 93, 38, 50, 62, 74, 86, 36, 48, 60, 72, 84, 96], # FADC
        "D7": [42, 54, 66, 78, 90, 46, 58, 70, 82, 94, 38, 50, 62, 74, 86, 36, 48, 60, 72, 84, 96], 
        "C7": [36, 48, 60, 72, 84, 96, 40, 52, 64, 76, 88, 43, 55, 67, 79, 91, 46, 58, 70, 82, 94], 
    }
    num_outside = 0
    num_inside = 0
    for i in range(onsets.shape[0]):
        if notes[i, 0] == 0:
            continue
        curr_chord = harmony[int(onsets[i])]
        good_notes = chord_profiles[curr_chord]
        curr_pitch = int(notes[i, 0])
        if curr_pitch in good_notes:
            num_inside += 1
        else:
            num_outside += 1

    return num_inside / notes.shape[0]
        
    
print(inside_outside_metric_rc(np.zeros((10, 2))))
    