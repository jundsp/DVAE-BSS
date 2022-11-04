#%%
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torch import nn
import torchaudio
import soundfile as sf
import torch
import torch.utils.data
from torchvision import datasets, transforms
import random
import pandas as pd
import matplotlib.pyplot as plt
from utils.utils import mag2db, db2mag
import torch.nn.functional as F

def walk_files(root, suffix, prefix=False, remove_suffix=False):
    root = os.path.expanduser(root)
    for dirName, subdirList, fileList in os.walk(root):
        for fname in fileList:
            if fname.lower().endswith(suffix):
                descr = fname[: -len(suffix)]
                if remove_suffix:
                    fname = fname[: -len(suffix)]
                if prefix:
                    fname = os.path.join(root, dirName,fname)
                yield fname, descr

class Audioset(Dataset):
    def __init__(self,root,duration=4,normalize=True,suffix='.wav',samplerate=16000,nsrc=1,transform=None,augment_levels=True,silence=.2):
        self._path = root
        self.duration = duration
        self.normalize = normalize
        self.suffix = suffix
        self.samplerate = samplerate
        self.samples = int(duration*samplerate)
        self.augment_levels = augment_levels

        walker = walk_files(self._path, suffix=suffix, prefix=True)
        self._walker = list(walker)

        self.n_files = len(self._walker)
        self.nsrc = nsrc
        self.transform = transform
        self.normalize = normalize
        self.silence_samples = int(silence*samplerate)+1
        self.dbmin = 30
        
    def _load_audio(self,n):
        path = self._walker[n][0]
        
        waveform, samplerate = torchaudio.load(os.path.join(self._path,path))
        if waveform.ndim > 1:
            waveform = waveform[0]
            
        if samplerate != self.samplerate:
            try:
                waveform = torchaudio.functional.resample(waveform,orig_freq=samplerate,new_freq=self.samplerate)
            except:
                print("Could not resample!")
                print(path)
        if self.normalize:
            waveform /= waveform.abs().max()

        if self.augment_levels:
            waveform *= db2mag(torch.rand(1)*self.dbmin-self.dbmin)
            
        return waveform
    
    def _mix_audio(self,x):
        if self.augment_levels:
            alpha = torch.rand(self.nsrc,device=x.device,dtype=x.dtype)
            alpha /= alpha.sum()
            x = x * alpha.view(-1,1)
        return x.sum(0), x
        
                
    def __getitem__(self, n):
        L = np.random.randint(self.silence_samples)
        k = n
        output = torch.zeros(self.samples)
        while L < self.samples:
            waveform = self._load_audio(k)
            N = len(waveform)
            if (N+L) >= self.samples:
                N = self.samples - L
            output[L:L+N] = waveform[:N]
            L += N
            L += np.random.randint(self.silence_samples)
            k = np.random.randint(self.n_files)
        
        return output

    def __len__(self):
        return self.n_files



class Mixer(torch.nn.Module):
    def __init__(
        self,
        n_sources=1,
        average=True,
    ):
        super(Mixer,self).__init__()
        self.n_sources = n_sources
        self.average = average

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.n_sources == 1:
            return waveform, waveform.unsqueeze(1)
        else:
            n_batch = waveform.shape[0]
            perms = torch.roll(torch.arange(n_batch),1)
            sources = torch.stack([waveform,waveform[perms]],1)
            if self.average:
                sources /= sources.shape[1]
            mix = sources.sum(1)
            return mix, sources


class AudioPipeline(torch.nn.Module):
    def __init__(
        self,
        n_fft=512,
        n_time=None,
        hop_length=256,
        stretch_factor=0.8,
        compression=1.0,
    ):
        super().__init__()
        
        self.n_time = n_time
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=1,normalized=True, hop_length=hop_length)
        self.compression = compression

        self.spec_aug = torch.nn.Sequential(
            torchaudio.transforms.TimeStretch(stretch_factor, fixed_rate=True),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
            torchaudio.transforms.TimeMasking(time_mask_param=10),
        )
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.spec(waveform) ** self.compression
        if self.n_time is not None:
            T = spec.shape[-1]
            if self.n_time < T:
                spec = spec.transpose(0,-1)[:self.n_time].transpose(0,-1)
            else:
                spec = F.pad(spec,(0,self.n_time-T))
        return spec

class Shift(object):
    def __init__(self, shift_amount=0.0):
        shift_amount = float(shift_amount)
        shift_amount = np.minimum(1.0,shift_amount)
        shift_amount = np.maximum(0.0,shift_amount)
        self.shift_amount = shift_amount

    def __call__(self, sample):
        
        if sample.ndim > 1:
            sample = sample.transpose(0,-1)
            
        max_shift = int(self.shift_amount * sample.shape[0])
        tau = np.random.randint(max_shift+1)
        direction = np.random.randint(2)
        out = sample.clone()
        if tau > 0:
            if direction == 1:
                out[tau:] = sample[:-tau]
                out[:tau] = 0
            elif direction == 0:
                out[:-tau] = sample[tau:]
                out[-tau:] = 0
            
        if sample.ndim > 1:
            out = out.transpose(0,-1)
        return out
    
class Crop(object):
    def __init__(self, size=None):
        self.size = size

    def __call__(self, sample):
        if self.size is not None:
            L = sample.shape[-1]
            if L < self.size:
                sample = nn.functional.pad(sample,(0,self.size-L))
            else:
                sample = sample.transpose(0,-1)[:self.size].transpose(0,-1)
        return sample
    
class Chain(object):
    def __init__(self,size=None) -> None:
        self.size = size
        
    def __call__(self,input):
        if self.size is not None:
            n = input.shape[0]
            
            k = self.size // input.shape[-1]
            m = n // k
            size_in = k*input.shape[-1]
            data = torch.zeros(m,input.shape[-2],self.size,dtype=input.dtype, layout=input.layout, device=input.device)
            for mm in range(m):
                idx = mm*k
                data[mm,:,:size_in] = torch.cat([input[i+idx] for i in range(k)],-1)
        else:
            data = input
                            
        return data


#%%
torch.manual_seed(1)
np.random.seed(1)
dataset = Audioset(root='/Users/julian/Desktop/RWC_small',duration=8,silence=.2,augment_levels=False)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True)

#%%
x = next(iter(dataloader))

#%%
mixer = Mixer(n_sources=2)
pipeline = AudioPipeline(n_fft=512,hop_length=128,compression=1,n_time=120)
x,sources = mixer(x)
X = pipeline(x)
print(X.shape)
print(X.max())
print(x.abs().max())
#%%
fig,ax = plt.subplots(2,2,figsize=(8,5))
for i in range(4):
    ax[i%2, i//2].imshow(mag2db(X[i]),aspect='auto',origin='lower',cmap='jet')

# %%
torchaudio.save('/Users/julian/Desktop/rwc_example_loaded.wav',x[3].view(1,-1),16000)
# %%
