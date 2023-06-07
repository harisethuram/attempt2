import torch
import numpy as np
import sqlite3
import math
from repalt import make_batch

def inside_outside_metric_rc(notes, start_offset=0, chorus_count=1):
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
    song_length = 128
    harmony = np.tile(harmony, 4)
    # print(notes.shape)
    onsets = np.floor(np.roll(np.cumsum(notes[:, 1]), shift=1))# beats in which notes begin
    
    onsets[0] = 0
    # print("N", notes[:20, :])
    # print(onsets[:20])
    befores = np.where(onsets < start_offset)
    onsets = np.delete(onsets, befores) - start_offset
    if np.array([befores]).size > 0:
        notes = notes[np.max(befores)+1:, :]
    # notes = np.delete(notes, befores)
    # onsets = onsets[i:] #- start_offset
    extras = np.where(onsets >= song_length * chorus_count)
    # print("e", notes[extras, :])
    # print(onsets[extras])
    onsets = np.delete(onsets, extras)
    print(notes[:10])
    print(onsets[:10])
    print(onsets[:10])
    notes[:, 1] = notes[:, 1]
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
    cur_beat = 0
    # i = 0
    # while cur_beat < 128+start_offset:
    #     if notes[i, 0] == 0 or start_offset > cur_beat:
    #         print("zero", cur_beat, notes[i, 0], notes[i, 1])
    #         cur_beat += notes[i, 1]
    #         i += 1
    #         continue
    #     curr_chord = harmony[math.floor(cur_beat) - start_offset]
    #     curr_note = notes[i, 0]
    #     good_notes = chord_profiles[curr_chord]
    #     if curr_note in good_notes:
    #         print("YES", curr_note, curr_chord, cur_beat)
    #         num_inside += 1
    #     else:
    #         print("NO", curr_note, curr_chord, cur_beat)
    #     cur_beat += notes[i, 1]
        # i += 1
        
        
    # print(harmony.shape)
    print(np.max(onsets))
    for i in range(onsets.shape[0]):
        if notes[i, 0] == 0:
            continue
        curr_chord = harmony[int(onsets[i])]
        # print(onsets[i])
        good_notes = chord_profiles[curr_chord]
        curr_pitch = int(notes[i, 0])
        # print(curr_chord, curr_pitch, onsets[i] % 4, np.floor(onsets[i]/4))
        if curr_pitch in good_notes:
            print("YES", curr_pitch, curr_chord, onsets[i])
            num_inside += 1
        else:
            print("NO", curr_pitch, curr_chord, onsets[i])
            num_outside += 1

    return num_inside / notes.shape[0]
        
def eval_songs_preprocess():
    con = sqlite3.connect("wjazzd.db")
    cur = con.cursor()
    preciseness = 24 # to what-th note will I be rounding
    melody_x = cur.execute("SELECT m.melid, m.pitch, m.onset, m.duration, m.beat, m.tatum, m.division, m.beatdur, s.avgtempo from melody m \
                           JOIN solo_info s ON m.melid=s.melid \
                           JOIN composition_info c ON s.compid=c.compid \
                           WHERE (c.template='I Got Rhythm' and s.key='Bb-maj')")
    melody = np.array(melody_x.fetchall())
    melids = np.unique(melody[:, 0])
    # batch songs
    melody_b = make_batch(melody, mode="mel")
    
    # quantize songs
    melody_q = np.zeros((melody_b.shape[0], melody_b.shape[1]*2, 2)) # pitch duration (pitch = 0 is rest)
    beatdurs = melody_b[:, :, -2]
    avgtempo = melody_b[:, :, -1]
    first_beat = melody_b[:, 0, 3]
    first_tatum = melody_b[:, 0, 4]
    first_division = melody_b[:, 0, 5]
    pitches = melody_b[:, :, 0]
    first_start_offset = (first_beat-1) + (first_tatum - 1) / first_division
    durations = np.nan_to_num(melody_b[:, :, 2] / (beatdurs), 0) # duration of every note
    durations_q = np.round(durations*preciseness)/preciseness

    onsets = np.nan_to_num(np.maximum(melody_b[:, :, 1] - np.expand_dims(melody_b[:, 0, 1], axis=1), 0) / (60/avgtempo), 0)
    onsets_q = np.round(onsets*preciseness)/preciseness + np.expand_dims(first_start_offset, axis=1)

    sum_thing = np.nan_to_num(onsets_q + durations_q, 0) # at what beat each note ends
    rests = np.round(np.maximum(onsets_q - np.roll(sum_thing, shift=1, axis=1), 0)*preciseness)/preciseness
    rests = np.round(np.maximum(onsets_q - np.roll(sum_thing, shift=1, axis=1), 0)*preciseness)/preciseness 
    
    rests = np.nan_to_num(rests, 0)
    durations_q = np.nan_to_num(durations_q, 0)
    
    melody_q[:, ::2, 1] = rests
    melody_q[:, 1::2, 1] = durations_q
    melody_q[:, 1::2, 0] = pitches

    non_zeros = np.zeros_like(melody_q)
    thing = np.copy(melody_q[:, :, 1])
    inds = np.where(thing != 0)
    curs = np.zeros((melody_q.shape[0])).astype(int)

    for i in range(len(inds[0])):
        non_zeros[inds[0][i], curs[inds[0][i]], :] = melody_q[inds[0][i], inds[1][i], :]
        curs[inds[0][i]] += 1
    
    # dont need none of that
    melody = non_zeros
    

    melody_dict = {melids[i] : melody[i, :, :] for i in range(melody.shape[0])}
    return melody_dict

def get_offsets():
    offsets = {'1': 4, '35': 7, '66': 0, '67': 5, '78': 2, '130': 8, '133': 0, '159': 0, '231': 0, '283': 16, '320': 0, '321': 0, 
                '360': 0, '365': 0}
    return offsets
def get_chorus_counts():
    con = sqlite3.connect("wjazzd.db")
    cur = con.cursor()
    c_count_x = cur.execute("SELECT chorus_count FROM solo_info s\
                           JOIN composition_info c ON s.compid=c.compid \
                           WHERE (c.template='I Got Rhythm' and s.key='Bb-maj')")
    c_count = np.array(c_count_x.fetchall())
    return c_count
    
def calc_avg():
    songs_melody = eval_songs_preprocess()
    offsets = get_offsets()
    c_count = get_chorus_counts()
    total = 0
    i = 0
    for mels in songs_melody:
        x = inside_outside_metric_rc(songs_melody[mels], offsets[str(int(mels))])
        print(x)
        total += x
        i += 1
        
    print(total / c_count.shape[0], c_count.shape[0])
    # print(inside_outside_metric_rc())
calc_avg()