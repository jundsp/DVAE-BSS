#%%
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torch import nn
import torchaudio
import soundfile as sf

def round_pow2(x):
    return int(np.power(2,np.ceil(np.log2(x))))

import torch
import torch.utils.data
from torchvision import datasets, transforms


def get_data_loaders(directory,batch_size,kwargs):
    train_set = datasets.MNIST(directory, train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST(directory, train=False, download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader
    
class STFT(nn.Module):
    def __init__(self, n_fft, hop_length = None):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft//4
        if self.hop_length is not None:
            self.hop_length = hop_length
            
        self.register_buffer("window",torch.hann_window(n_fft))

    def forward(self, x):
        X = torch.stft(x, n_fft = self.n_fft, hop_length=self.hop_length, win_length=None, window=self.window, center=True, pad_mode='reflect', normalized=False, onesided=True,return_complex=False)
        return X

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
    def __init__(self,root,duration=None,normalize=True,suffix='.wav',bins=None,spectrogram=False):
        self._path = root
        self.duration = duration
        self.normalize = normalize
        self.suffix = suffix
        self.spectrogram = spectrogram
        self.bins = bins

        walker = walk_files(self._path, suffix=suffix, prefix=True)
        self._walker = list(walker)
        self.n_files = len(self._walker)
        # self.n_files = np.minimum(self.n_files,1024)
        
        print(root)
        
        self.resampler = torchaudio.transforms.Resample(orig_freq=44100,new_freq=16000)
        self.stft = STFT(n_fft=256,hop_length=128)

    def __getitem__(self, n):
        path, descr = self._walker[n]

        waveform, samplerate = torchaudio.load(path)
        if waveform.ndim > 1:
            waveform = waveform[0]
        L = len(waveform)
        starting = np.random.randint(0,L//4)
        ending = starting + 44100*4
        waveform = self.resampler(waveform[starting:ending])
        if self.spectrogram is True:
            data = self.process_spectrogram(waveform)
        else:
            data = self.process_audio(waveform)
            
        return data, 16000

    def process_audio(self,waveform):
        if self.normalize is True:
            waveform /= waveform.abs().max()
        if self.duration is not None:
            temp = waveform
            waveform = torch.zeros(1,self.duration)
            n = min(temp.size(-1),self.duration)
            waveform[:,:n] = temp[:,:n]
        return waveform

    def process_spectrogram(self,waveform):
        data = torch.norm(self.stft(waveform),2,-1).squeeze()
        if self.normalize is True:
            normalizer = data.abs().max()
            if normalizer == 0: normalizer = 1.0
            data /= normalizer
        if self.duration is not None:
            out = torch.zeros(self.bins,self.duration)
            if data.shape[-1] < self.duration:
                out[:,:data.shape[-1]] = data[:self.bins]
            else:
                out = data[:self.bins,:self.duration]
            data = out + 0
        return data.unsqueeze(0)

    def __len__(self):
        return self.n_files


class MUMS(Dataset):
    def __init__(self,root,duration=None,normalize=True,bins=None,train=True,magnitude=False):
        if train is True:
            self._path = os.path.join(root,'stft/train')
        else:
            self._path = os.path.join(root,'stft/test')
        self.duration = duration
        self.normalize = normalize
        self.bins = bins
        self.magnitude = magnitude
        suffix = '.pt'

        walker = walk_files(self._path, suffix=suffix, prefix=True)
        self._walker = list(walker)
        self.n_files = len(self._walker)

    def __getitem__(self, n):
        path, descr = self._walker[n]

        data, fname,_ = torch.load(path)
        if self.normalize is True:
            data /= torch.norm(data,2,-1).abs().max()
        if self.magnitude is True:
            data = torch.norm(data,2,-1)        
        data = data.squeeze()[:self.bins,:self.duration].unsqueeze(0)
        return data.clone(), fname

    def __len__(self):
        return self.n_files
    
    
class WHAMR(Dataset):
    def __init__(self,root,duration=None,set='train'):
        
        sets = {'train': 'tr',
                'validate': 'cv',
                'test': 'tt'}
        self._path = os.path.join(root,'whamr/wav16k/min/'+sets[set])
        self.duration = duration
        suffix = '.wav'
        
        path_input = os.path.join(self._path,"s1_anechoic")

        walker = walk_files(path_input, suffix=suffix, prefix=True)
        self._walker = list(walker)
        self.n_files = len(self._walker)

    def __getitem__(self, n):
        path, filename = self._walker[n]

        x,fs = torchaudio.load(path)

        if self.duration is not None:
            L = x.shape[-1]
            samples = int(self.duration * fs)
            if samples > L:
                x = torch.cat([x,torch.zeros((1,samples-L))],-1)
            x = x[:,:samples]

        return x[0]

    def __len__(self):
        return self.n_files

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    dataset = Audioset(root='/home/jneri/projects/def-depalle/jneri/data/MUMS',duration=4)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=True)
    x = next(iter(dataloader))
        
    X = torch.stft(x[0],n_fft=512,return_complex=True,window=torch.hann_window(512))    
    plt.imshow(X.abs().log10(),aspect='auto',origin='lower',cmap='jet')
    
    

# %%
