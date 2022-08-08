#%%
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from torch import nn
import torchaudio

def round_pow2(x):
    return int(np.power(2,np.ceil(np.log2(x))))

class STFT(nn.Module):
    def __init__(self, n_fft, hop_length = None):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft//4
        if self.hop_length is not None:
            self.hop_length = hop_length
            
        self.register_buffer("window",torch.hann_window(n_fft))

    def forward(self, x):
        X = torch.stft(x, n_fft = self.n_fft, hop_length=self.hop_length, win_length=None, window=self.window, center=True, pad_mode='reflect', normalized=False, onesided=True)
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

    def __getitem__(self, n):
        path, descr = self._walker[n]

        waveform, samplerate = sf.load(path)
        
        if self.spectrogram is True:
            n_fft = round_pow2(samplerate*.025)
            data = self.process_spectrogram(waveform,n_fft)
        else:
            data = self.process_audio(waveform)

        return data, samplerate

    def process_audio(self,waveform):
        if self.normalize is True:
            waveform /= waveform.abs().max()
        if self.duration is not None:
            temp = waveform
            waveform = torch.zeros(1,self.duration)
            n = min(temp.size(-1),self.duration)
            waveform[:,:n] = temp[:,:n]
        return waveform

    def process_spectrogram(self,waveform,n_fft):
        data = torch.norm(STFT(n_fft=n_fft)(waveform),2,-1).squeeze()
        if self.normalize is True:
            data /= data.abs().max()
        if self.duration is not None:
            data = data[:self.bins,:self.duration].unsqueeze(0)
        return data

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
    
    dataset = WHAMR(root='/home/jneri/scratch/data/',duration=4)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=True)
    x = next(iter(dataloader))
        
    X = torch.stft(x[0],n_fft=512,return_complex=True,window=torch.hann_window(512))    
    plt.imshow(X.abs().log10(),aspect='auto',origin='lower',cmap='jet')
    
    

# %%
