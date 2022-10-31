
import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.signal as sig
import os

input_path = Path('/Users/julian/Desktop/')
output_path = Path('/Users/julian/Documents/Desktop')

# filename = '073ACR1F'
filename = '011PFREF'
suffix = '.wav'
track_file = input_path.joinpath(filename)
x,fs = sf.read(track_file.parent / (track_file.name + suffix))

y = np.abs(x)
y /= y.max()

L = 500
b = np.ones(L) / L
y = sig.lfilter(b,1, y)

min_length = .1*fs

ydb = 20*np.log10(y)
thresh = -100
low = (ydb < thresh).astype(int)
low[0] = 1
low[-1] = 1
changepoints = np.diff(low)
turnon = np.nonzero(changepoints == -1)[0]
turnoff = np.nonzero(changepoints == 1)[0]

split = []
os.makedirs(track_file,exist_ok=True)
for i in range(len(turnon)):
    if turnoff[i] - turnon[i] > min_length:
        stem = x[turnon[i]:turnoff[i]]
        sf.write(file=track_file.joinpath(filename + '_{}'.format(i+1) + suffix),data=stem,samplerate=fs)

        split.append(stem)



plt.plot(x)
plt.plot(y)
plt.plot(low)
plt.xlim(0,1e6)