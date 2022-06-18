"""
Reformat audio files to (sox required):
    - 16000 Hz
    - mono
    - 16-bit
"""
import os
import sys
from tqdm import tqdm
from glob import glob

DATA_PATH = sys.argv[1]
filenames = glob(DATA_PATH + '/*/*/*.wav')
for filename in tqdm(filenames):
    os.system(f'sox {filename} -r 16k -c 1 -b 16 temp.wav')
    os.system(f'mv temp.wav {filename}')
print('Done!')
