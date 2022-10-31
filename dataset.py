#%%
from locale import normalize
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

def round_pow2(x):
    return int(np.power(2,np.ceil(np.log2(x))))

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
    def __init__(self,root,path_csv,duration=4,normalize=True,suffix='.wav',samplerate=16000,nsrc=2,transform=None):
        self._path = root
        self.duration = duration
        self.normalize = normalize
        self.suffix = suffix
        self.samplerate = samplerate
        self.samples = int(duration*samplerate)
        self._walker = pd.read_csv(os.path.join(root,path_csv),header=None)[0].tolist()

        # walker = walk_files(self._path, suffix=suffix, prefix=True)
        # self._walker = list(walker)

        self.n_files = len(self._walker)
        self.nsrc = nsrc
        self.transform = transform
        
    def _load_audio(self,n):
        path = self._walker[n]
        
        waveform, samplerate = torchaudio.load(os.path.join(self._path,path))
        if waveform.ndim > 1:
            waveform = waveform[0]
        
        if samplerate != self.samplerate:
            waveform = torchaudio.functional.resample(waveform,orig_freq=samplerate,new_freq=self.samplerate)
        if normalize:
            waveform /= waveform.abs().max()
                        
        data = torch.zeros(self.samples)
        if len(waveform) < self.samples:
            data[:len(waveform)] = waveform
        else:
            data = waveform[:self.samples]
        
        if self.transform:
            data = self.transform(data)
            
        return data
    
    def _mix_audio(self,x):
        return x.sum(0)
        
                
    def __getitem__(self, n):
        k = random.sample(range(0, self.n_files), self.nsrc)
        if n not in k:
            k[0] = n
        
        out = []
        for i in range(self.nsrc):
            out.append(self._load_audio(k[i]))
        
        sources = torch.stack(out,0)
        mix = self._mix_audio(sources)
        
        return mix

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

class AudioPipeline(torch.nn.Module):
    def __init__(
        self,
        n_fft=512,
        n_time=256,
        hop_length=256,
        stretch_factor=0.8,
        compression=1.0,
    ):
        super().__init__()
        
        self.affine = transforms.RandomAffine(degrees=0,translate=(0.0,0.0))
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=1,normalized=True, hop_length=hop_length)
        self.compression = compression

        self.spec_aug = torch.nn.Sequential(
            torchaudio.transforms.TimeStretch(stretch_factor, fixed_rate=True),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
            torchaudio.transforms.TimeMasking(time_mask_param=10),
        )
        
        self.crop = Crop(size=n_time)
        


    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.spec(waveform) ** self.compression
        spec = self.crop(spec)
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
        max_shift = self.shift_amount * sample.shape[0]
        tau = np.random.randint(int(max_shift)+1)
        out = sample.clone()
        if tau > 0:
            out[tau:] = sample[:-tau]
            out[:tau] = 0.0
        if sample.ndim > 1:
            out = out.transpose(0,-1)
        return out
    
class Crop(object):
    def __init__(self, size):
        assert isinstance(size, (int))
        self.size = size

    def __call__(self, sample):
        L = sample.shape[-1]
        if L < self.size:
            sample = nn.functional.pad(sample,(0,self.size-L))
        else:
            sample = sample.transpose(0,-1)[:self.size].transpose(0,-1)
        return sample
    
#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    transform = transforms.Compose([Shift(shift_amount=.1)])
    dataset = Audioset(root='/home/jneri/projects/def-depalle/jneri/data',duration=8,transform=transform,path_csv='rwc_dataset/test.csv')
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=True)
    #%%
    x = next(iter(dataloader))
    #%%
    pipeline = AudioPipeline(n_time=256,hop_length=128,compression=.5)
    X = pipeline(x)
    #%%
    plt.imshow(X[11],aspect='auto',origin='lower',cmap='jet')
    
# %%
