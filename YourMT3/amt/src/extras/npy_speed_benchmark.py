import os
from tasks.utils.event_codec import Event, EventRange
from tasks.utils import event_codec

ec = event_codec.Codec(
            max_shift_steps=1000,  # this means 0,1,...,1000
            steps_per_second=100,
            event_ranges=[
                EventRange('pitch', min_value=0, max_value=127),
                EventRange('velocity', min_value=0, max_value=1),
                EventRange('tie', min_value=0, max_value=0),
                EventRange('program', min_value=0, max_value=127),
                EventRange('drum', min_value=0, max_value=127),
            ],
        )

events = [
    Event(type='shift', value=0),  # actually not needed
    Event(type='shift', value=1),  # 10 ms shift
    Event(type='shift', value=1000),  # 10 s shift
    Event(type='pitch', value=0),  # lowest pitch 8.18 Hz
    Event(type='pitch', value=60),  # C4 or 261.63 Hz
    Event(type='pitch', value=127),  # highest pitch G9 or 12543.85 Hz
    Event(type='velocity', value=0),  # lowest velocity)
    Event(type='velocity', value=1),  # lowest velocity)
    Event(type='tie', value=0),  # tie
    Event(type='program', value=0),  # program
    Event(type='program', value=127),  # program
    Event(type='drum', value=0),  # drum
    Event(type='drum', value=127),  # drum
]

events = events * 100
tokens = [ec.encode_event(e) for e in events]
tokens = np.array(tokens, dtype=np.int16)

import csv
# Save events to a CSV file
with open('events.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for event in events:
        writer.writerow([event.type, event.value])

# Load events from a CSV file
with open('events.csv', 'r') as file:
    reader = csv.reader(file)
    events2 = [Event(row[0], int(row[1])) for row in reader]


import json
# Save events to a JSON file
with open('events.json', 'w') as file:
    json.dump([event.__dict__ for event in events], file)

# Load events from a JSON file
with open('events.json', 'r') as file:
    events = [Event(**event_dict) for event_dict in json.load(file)]




"""----------------------------"""
# Write the tokens to a npy file
import numpy as np
np.save('tokens.npy', tokens)

def t_npy():
    t = np.load('tokens.npy', allow_pickle=True) # allow pickle doesn't affect speed

os.makedirs('temp', exist_ok=True)
for i in range(2400):
    np.save(f'temp/tokens{i}.npy', tokens)

def t_npy2400():
    for i in range(2400):
        t = np.load(f'temp/tokens{i}.npy')
def t_npy2400_take200():
    for i in range(200):
        t = np.load(f'temp/tokens{i}.npy')

import shutil
shutil.rmtree('temp', ignore_errors=True)

# Write the 2400 tokens to a single npy file
data = dict()
for i in range(2400):
    data[f'arr{i}'] = tokens.copy()
np.save(f'tokens_2400x.npy', data)
def t_npy2400single():
    t = np.load('tokens_2400x.npy', allow_pickle=True).item()

def t_mmap2400single():
    t = np.load('tokens_2400x.npy', mmap_mode='r')

# Write the tokens to a npz file
np.savez('tokens.npz', arr0=tokens)
def t_npz():
    npz_file = np.load('tokens.npz')
    tt = npz_file['arr0']

data = dict()
for i in range(2400):
    data[f'arr{i}'] = tokens
np.savez('tokens.npz', **data )
def t_npz2400():
    npz_file = np.load('tokens.npz')
    for i in range(2400):
        tt = npz_file[f'arr{i}']

def t_npz2400_take200():
    npz_file = np.load('tokens.npz')
    # npz_file.files
    for i in range(200):
        tt = npz_file[f'arr{i}']


# Write the tokens to a txt file
with open('tokens.txt', 'w') as file:
    file.write(' '.join(map(str, tokens)))

def t_txt():
    # Read the tokens from the file
    with open('tokens.txt', 'r') as file:
        t = list(map(int, file.read().split()))
    t = np.array(t)


# Write the tokens to a CSV file
with open('tokens.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(tokens)

def t_csv():
    # Read the tokens from the CSV file
    with open('tokens.csv', 'r') as file:
        reader = csv.reader(file)
        t = list(map(int, next(reader)))
        t = np.array(t)


# Write the tokens to a JSON file
with open('tokens.json', 'w') as file:
    json.dump(tokens, file)

def t_json():
    # Read the tokens from the JSON file
    with open('tokens.json', 'r') as file:
        t = json.load(file)
        t = np.array(t)

with open('tokens_2400x.json', 'w') as file:
    json.dump(data, file)

def t_json2400single():
    # Read the tokens from the JSON file
    with open('tokens_2400x.json', 'r') as file:
        t = json.load(file)      

def t_mmap():
    t = np.load('tokens.npy', mmap_mode='r')

# Write the tokens to bytes file




np.savetxt('tokens.ntxt', tokens)
def t_ntxt():
    t = np.loadtxt('tokens.ntxt').astype(np.int32)

%timeit t_npz() # 139 us
%timeit t_mmap() # 3.12 ms 
%timeit t_npy() # 87.8 us
%timeit t_txt() # 109 152 us
%timeit t_csv() # 145 190 us
%timeit t_json() # 72.8 119 us
%timeit t_ntxt() # 878 us

%timeit t_npy2400() # 212 ms; 2400 files in a folder
%timeit t_npz2400() # 296 ms; uncompreesed 1000 arrays in a single file

%timeit t_npy2400_take200() # 17.4 ms; 25 Mb
%timeit t_npz2400_take200() # 28.8 ms; 3.72 ms for 10 arrays; 25 Mb
%timeit t_npy2400single() # 4 ms; frozen dictionary containing 2400 arrays; 6.4 Mb; int16
%timeit t_mmap2400single() # dictionary is not supported 
%timeit t_json2400single() # 175 ms; 17 Mb
# 2400 files from 100ms hop for 4 minutes   