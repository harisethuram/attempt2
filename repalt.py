import torch
import numpy as np
import sqlite3
import os.path
import pickle
from generatemidi import generate_midi

# harmony is the same
def get_harmony():
    """
    - returns filtered and augmented harmony (N, 2) i.e. with only songs in 4/4 and the uncommon chords replaced with 'unk'
    """
    con = sqlite3.connect("wjazzd.db")
    cur = con.cursor()

    # only select 4/4
    harmony_x = cur.execute("SELECT b.melid, b.chord from beats b JOIN solo_info s ON b.melid=s.melid WHERE (s.signature='4/4' AND s.chorus_count<8)")
    harmony = np.array(harmony_x.fetchall())

    # get max melid
    max_melid = np.max(harmony[:, 0].astype(int))

    # get num songs
    num_songs = np.unique(harmony[:,0]).shape[0]
    num_keys = 12
    
    # get num events
    num_events = harmony.shape[0]

    # fill in blank spaces
    prev_tok = "start"
    cur_melid = '1'
    for i in range(harmony.shape[0]):
        if harmony[i, 0] != cur_melid:
            cur_melid = harmony[i, 0]
            if harmony[i,1] == '':
                prev_tok = "start"
            else:
                prev_tok = harmony[i,1]
        
        if harmony[i,1] != '':
            prev_tok = harmony[i,1]
        else:
            harmony[i,1] = prev_tok
    
    # batch
    harmony_b = make_batch(harmony, mode="har")
    # print(harmony_b[1, :20, 0])

    # augment harmony
    harmony_aug = np.full((harmony_b.shape[0]*num_keys, harmony_b.shape[1], 1), '').astype(object)

    for i in range(harmony_b.shape[0]):
        for j in range(harmony_b.shape[1]):
            augs = None
            if harmony_b[i, j, 0] != '' and harmony_b[i, j, 0] != 'NC' and harmony_b[i, j, 0] != 'start':
                # print(harmony_b[i, j, 0])
                augs = augment_harmony(harmony_b[i, j, 0])
            elif harmony_b[i, j, 0] == 'start':
                augs = np.tile(np.array(['start']), num_keys)
            elif harmony_b[i, j, 0] == '':
                break
            elif harmony_b[i, j, 0] == 'NC':
                augs = np.tile(np.array(['unk']), num_keys)
            
            # get indices to modify
            indices = np.arange(0, stop=num_keys, step=1) * harmony_b.shape[0] + i
            harmony_aug[indices, j, 0] = augs
    # print(harmony_aug[0, :20, 0])
    # print(harmony_aug[435*11, :20, 0])

    # don't need none of that
    harmony = harmony_aug

    # frequencies
    harmony_vocab, counts = np.unique(harmony[:, :, 0], return_counts=True)
    freqs = np.array((harmony_vocab, counts)).T
    freqs_dict = {row[0]: int(row[1]) for row in freqs}

    # unks
    for i in range(harmony.shape[0]):
        for j in range(harmony.shape[1]):
            if harmony[i, j, 0] == '':
                break
            if freqs_dict[harmony[i, j, 0]] < 20:
                harmony[i, j, 0] = 'unk'
    harmony_vocab, counts = np.unique(harmony[:, :, 0], return_counts=True)
    freqs = np.array((harmony_vocab, counts)).T
    freqs_dict = {row[0]: int(row[1]) for row in freqs}

    # print(freqs_dict)
    print(harmony.shape)
    return harmony_vocab, harmony


def augment_harmony(chord):
    """
    - take in string representing chord
    - returns all 12 transpositions of chord
    """
    num_keys = 12
    transpose = {
        'A': ['A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab'],
        'A#': ['Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A'],
        'Bb': ['Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A'],
        'B': ['B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb'],
        'Cb': ['B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb'],
        'C': ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'],
        'B#': ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B'],
        'Db': ['Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C'],
        'C#': ['Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C'],
        'D': ['D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db'],
        'Eb': ['Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D'],
        'D#': ['Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D'],
        'E': ['E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb'],
        'Fb': ['E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb'],
        'F': ['F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E'],
        'E#': ['F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E'],
        'F#': ['Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F'],
        'Gb': ['Gb', 'G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F'],
        'G': ['G', 'Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb'],
        'Ab': ['Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G'],
        'G#': ['Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G'],
    }
    # in case of slash chords
    chord = chord.split("/")
    upper = chord[0]
    bass = ''
    if len(chord) > 1:
        bass = chord[1]
    keys = transpose.keys()
    key=None
    fluff = '' # extensions and chord quality
    if upper[:2] in keys:
        key = upper[:2]
        fluff = upper[2:]
    elif upper[:1] in keys:
        key = upper[:1]
        fluff = upper[1:]
    key_b = ''
    fluff_b = ''
    if bass != '':
        if bass[:2] in keys:
            key_b = bass[:2]
            fluff_b = bass[2:]
        elif bass[:1] in keys:
            key_b = bass[:1]
            fluff_b = bass[1:]

    transpositions = []

    for i in range(num_keys):
        bass_transpose = ''
        if bass != '':
            bass_transpose = "/" + transpose[key_b][i] + fluff_b
        transpositions.append(transpose[key][i]+fluff+bass_transpose)
    return np.array(transpositions)

def get_melody():
    """
    - returns filtered harmony (N, 5) i.e. with only songs in 4/4 and the uncommon notes replaced with (0, 0, 0, 0)
    """
    con = sqlite3.connect("wjazzd.db")
    cur = con.cursor()
    # only select 4/4 
    melody_x = cur.execute("SELECT m.melid, m.pitch, m.onset, m.duration, m.beat, m.tatum, m.division, m.beatdur, s.avgtempo from melody m JOIN solo_info s ON m.melid=s.melid WHERE (s.signature='4/4' AND s.chorus_count<8)")
    melody = np.array(melody_x.fetchall()) # N,5, melid pitch onset duration beat tatum division beatdur
    preciseness = 24 # to what-th note will I be rounding
    # get max melid
    max_melid = np.max(melody[:, 0].astype(int))

    # get num songs
    num_songs = np.unique(melody[:,0]).shape[0]
    num_keys = 12

    # get num events
    num_events = melody.shape[0]

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

    # print(np.where(durations_q == np.max(durations_q)))
    # print(durations_q[300, 363])
    # print(beatdurs[300, 363])
    # print(melody_b[300, 363, 2])
    onsets = np.nan_to_num(np.maximum(melody_b[:, :, 1] - np.expand_dims(melody_b[:, 0, 1], axis=1), 0) / (60/avgtempo), 0)
    onsets_q = np.round(onsets*preciseness)/preciseness + np.expand_dims(first_start_offset, axis=1)
    # print("max1", np.max(onsets_q))
    # print(np.where(onsets_q == np.max(onsets_q)))
    # print(beatdurs[219, 4767])
    # print(np.nan_to_num(np.maximum(melody_b[:, :, 1] - np.expand_dims(melody_b[:, 0, 1], axis=1), 0)[219, 4767]))
    # print(melody_b[219, 4767, 1])
    # calculate rests
    sum_thing = np.nan_to_num(onsets_q + durations_q, 0) # at what beat each note ends
    rests = np.round(np.maximum(onsets_q - np.roll(sum_thing, shift=1, axis=1), 0)*preciseness)/preciseness # 
    # print(melody_b[219, 4085:4098, 1])
    # print(beatdurs[219, 4085:4098])
    # print(melody_b[219, 4085:4098, 1]/(60/avgtempo[219, 4085:4098]))
    # print(durations_q[108, 450: 458])
    # print(sum_thing[108, 450:458])
    # print(onsets_q[108, 450:458])
    # print(melody_b[108, 450:458, 1])
    # print(rests[108, 450:458])
    rests = np.round(np.maximum(onsets_q - np.roll(sum_thing, shift=1, axis=1), 0)*preciseness)/preciseness # 
    
    # remove all nans
    rests = np.nan_to_num(rests, 0)
    durations_q = np.nan_to_num(durations_q, 0)
    # print("max:", np.max(rests))
    # print(np.where(rests == np.max(rests)))
    melody_q[:, ::2, 1] = rests
    melody_q[:, 1::2, 1] = durations_q
    melody_q[:, 1::2, 0] = pitches
    zero_rests = np.where(melody_q[:, :, 1]==0)

    non_zero_mask = melody_q[:, :, 1] != 0

    # Count the number of non-zero elements in each sequence
    num_non_zeros = np.sum(non_zero_mask, axis=1)

    non_zeros = np.zeros_like(melody_q)
    thing = np.copy(melody_q[:, :, 1])
    inds = np.where(thing != 0)
    curs = np.zeros((melody_q.shape[0])).astype(int)

    for i in range(len(inds[0])):
        non_zeros[inds[0][i], curs[inds[0][i]], :] = melody_q[inds[0][i], inds[1][i], :]
        curs[inds[0][i]] += 1
    
    # dont need none of that
    melody = non_zeros

    # augment melody
    melody_aug = np.zeros((melody.shape[0]*num_keys, melody.shape[1], 2))
    for i in range(melody.shape[0]):
        for j in range(melody.shape[1]):
            if melody[i, j, 0] == 0 and melody[i, j, 1] == 0:
                break
            else:
                augs = augment_melody(melody[i, j, :])
                indices = np.arange(0, stop=num_keys, step=1) * melody.shape[0] + i
                melody_aug[indices, j, :] = augs
            
    melody = melody_aug
    flattened_melody = np.reshape(melody, (melody.shape[0]*melody.shape[1], melody.shape[2]))
    melody_vocab, counts = np.unique(flattened_melody, axis=0, return_counts=True)
    freqs = np.concatenate((melody_vocab, np.expand_dims(counts, axis=1)), axis=1)
    freqs_dict = {tuple(row[:2]): int(row[2]) for row in freqs}
    print(melody.shape, len(freqs_dict))
    return melody_vocab, melody


def augment_melody(note):
    """
    - takes in note (2, ) np array where first index is pitch
    - returns all 12 transpositions upwards an octave of the note (12, 4)
    """
    note = np.expand_dims(note, axis=1)
    transpositions = np.tile(note, 12, )
    if note[0] != 0:
        amount = np.arange(start=0, stop=12, step=1)
    else:
        amount = np.zeros((12))
    transpositions[0, :] += amount
    return transpositions.T

def make_batch(songs, mode):
    """
        - takes in songs in dataset in form (T, d+1) where T is total number of events, and d is representation dimension (+1 as zeroeth column is the song id)
        - returns them batched by songs (N, M, d) where N is num songs, M is max length of a song
    """
    id = songs[:, 0].astype(int)
    song_ids = np.unique(id)
    num_songs = song_ids.shape[0]

    max_length = np.max(np.bincount(songs[:, 0].astype(int)))
    lengths = np.bincount(songs[:, 0].astype(int))
    print(mode, np.average(lengths))
    print(mode, np.std(lengths))
    print(mode, max_length)
    dim = songs.shape[1] - 1

    batched = None
    if mode == "mel":
        batched = np.zeros((num_songs, max_length, dim))
    elif mode == "har":
        batched = np.full((num_songs, max_length, dim), "", dtype=object)

    for i in range(num_songs):
        song = np.squeeze(songs[np.where(id==song_ids[i]), 1:], axis=0)        
        batched[i, :song.shape[0]] = song
        
    return batched

def indices_har(songs, vocab):
    """
    - takes in batched harmony of songs in shape and vocab
    - returns their indices with dictionary
    """
    song_to_i = { s:i for i, s in enumerate(vocab) }
    
    i_to_song = { i:s for i, s in enumerate(vocab) }
    
    inds = np.zeros((songs.shape[0], songs.shape[1]))
    for i in range(songs.shape[0]):
        for j in range(songs.shape[1]):
            # print(songs[0, 0, 0])
            s = songs[i, j, 0]
            inds[i, j] = song_to_i[s]
    return inds, song_to_i, i_to_song

def indices_mel(songs, vocab):
    """
    - takes in batched melody of songs in shape and vocab
    - returns their indices with dictionary
    """
    song_to_i = { tuple(s):i for i, s in enumerate(vocab) }
    i_to_song = { i:s for i, s in enumerate(vocab) }

    inds = np.zeros((songs.shape[0], songs.shape[1]))
    for i in range(songs.shape[0]):
        for j in range(songs.shape[1]):
            # print(songs[0, 0, 0])
            s = tuple(songs[i, j, :])
            inds[i, j] = song_to_i[tuple(s)]
    
    return inds, song_to_i, i_to_song

def preprocess(reload=False):
    if ((not os.path.isfile('harmony.pt')) and (not os.path.isfile('melody.pt'))) or reload==True:
        print("Creating data")
        harmony_vocab, harmony = get_harmony()
        melody_vocab, melody = get_melody()
        
        ih, har_to_i, i_to_har = indices_har(harmony, harmony_vocab)
        im, mel_to_i, i_to_mel = indices_mel(melody, melody_vocab)
        # save values as files
        print("Saving")
        torch.save(im, 'melody.pt')
        torch.save(ih, 'harmony.pt')
        with open('har_to_i.pkl', 'wb') as fp:
            pickle.dump(har_to_i, fp)
        with open('i_to_har.pkl', 'wb') as fp1:
            pickle.dump(i_to_har, fp1)
        with open('mel_to_i.pkl', 'wb') as fp2:
            pickle.dump(mel_to_i, fp2)
        with open('i_to_mel.pkl', 'wb') as fp3:
            pickle.dump(i_to_mel, fp3)
    else:
        print("loading data")
        ih = torch.load('harmony.pt')
        im = torch.load('melody.pt')
        
        with open('har_to_i.pkl', 'rb') as fp:
            har_to_i = pickle.load(fp)
        with open('i_to_har.pkl', 'rb') as fp1:
            i_to_har = pickle.load(fp1)
        with open('mel_to_i.pkl', 'rb') as fp2:
            mel_to_i = pickle.load(fp2)
        with open('i_to_mel.pkl', 'rb') as fp3:
            i_to_mel = pickle.load(fp3)

    return ih, har_to_i, i_to_har, im, mel_to_i, i_to_mel
# print(ih.shape)
# print("********************************************************************")
# print(i_to_har)
# print("********************************************************************")
# print(im)
# print("********************************************************************")
# print(i_to_mel)
