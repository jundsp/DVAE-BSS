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
from utils.utils import mag2db, db2mag
import torch.nn.functional as F
import librosa
from pathlib import Path

def round_pow2(x):
    return int(np.power(2,np.ceil(np.log2(x))))

def get_data_loaders(directory,batch_size,kwargs):
    train_set = datasets.MNIST(directory, train=True, download=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST(directory, train=False, download=True, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_set,batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set,batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def get_datasets(dataset,path,args):
    if dataset == 'MNIST':
        dim_data = (33,32)
        transform = transforms.Compose([transforms.CenterCrop(size=dim_data),transforms.RandomAffine(degrees=0,translate=(0.05,0.2)),transforms.ToTensor()])
        train_set = datasets.MNIST(path, train=True, transform=transform, download=True)
        test_set = datasets.MNIST(path, train=False, transform=transform, download=True)

    if dataset == 'MUMS':
        train_set = Audioset(path + '/MUMS', path_csv='train.csv',augment_levels=args.augment_levels,duration=float(args.duration),silence=.5)
        test_set = Audioset(path + '/MUMS', path_csv='test.csv',augment_levels=args.augment_levels,duration=float(args.duration),silence=.5)
                
    if dataset == 'RWC':
        print(path)
        train_set = Audioset(root=path + '/RWC_dataset',path_csv='train.csv',augment_levels=args.augment_levels,duration=float(args.duration),silence=.5)
        test_set =  Audioset(root=path + '/RWC_dataset',path_csv='validate.csv',augment_levels=args.augment_levels,duration=float(args.duration),silence=.5)

    if dataset == 'VCTK':
        train_set = Audioset(root=path + '/VCTK',path_csv='train.csv',augment_levels=args.augment_levels,duration=float(args.duration),silence=.1,starting=1,ending=1)
        test_set =  Audioset(root=path + '/VCTK',path_csv='validate.csv',augment_levels=args.augment_levels,duration=float(args.duration),silence=.1,starting=1,ending=1)

    if dataset == 'composite':
        train_set = Audioset(root=path,path_csv='composite/train.csv',augment_levels=args.augment_levels,duration=float(args.duration),silence=.5,starting=.5)
        test_set =  Audioset(root=path,path_csv='composite/validate.csv',augment_levels=args.augment_levels,duration=float(args.duration),silence=.5,starting=.5)

    if dataset == 'nsynth':
        kwds_dset = dict(augment_levels=args.augment_levels,duration=float(args.duration),silence=.5,starting=.1,ending=.5,remove_silence=args.remove_silence)
        train_set = Audioset(root=path + '/nsynth-train',path_csv=None,**kwds_dset)
        test_set =  Audioset(root=path + '/nsynth-valid',path_csv=None,**kwds_dset)

    if dataset == 'MUSDB':
        train_set = Audioset(root=path + '/MUSDB',path_csv='train.csv',augment_levels=args.augment_levels,duration=float(args.duration),silence=0.0,starting=60)
        test_set =  Audioset(root=path + '/MUSDB',path_csv='validate.csv',augment_levels=args.augment_levels,duration=float(args.duration),silence=0.0,starting=60)

    return train_set, test_set
    
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
                yield fname

class Audioset(Dataset):
    def __init__(self,root,path_csv=None,duration=4,normalize=True,suffix='.wav',samplerate=16000,augment_levels=False,silence=.1,max_files=20,starting=0,ending=0,remove_silence=False):
        self._path = root
        self.duration = duration
        self.normalize = normalize
        self.suffix = suffix
        self.samplerate = samplerate
        self.samples = int(duration*samplerate)
        self.augment_levels = augment_levels

        starting = np.maximum(0.0,float(starting))
        ending = np.maximum(0.0,float(ending))
        silence = np.maximum(0.0, float(silence))

        if path_csv is None:
            walker = walk_files(self._path, suffix=suffix, prefix=True)
            self._walker = list(walker)
        else:
            
            # self._walker = pd.read_csv(os.path.join(root,path_csv))['path'].tolist()
            self._walker = pd.read_csv(os.path.join(root,path_csv),header=None,sep='\t')[0].tolist()
           

        self.n_files = len(self._walker)
        self.normalize = normalize
        self.silence_samples = int(silence*samplerate)+1
        self.starting_samples = int(starting*samplerate) + 1
        self.ending_samples = int(ending*samplerate) + 1
        self.dbmin = 30
        self.max_files = max_files
        self.remove_silence = remove_silence
        
    def _load_audio(self,n):
        path = self._walker[n]
        
        waveform, samplerate = torchaudio.load(os.path.join(self._path,path))
        if waveform.ndim > 1:
            waveform = waveform.mean(0)
            
        if samplerate != self.samplerate:
            print("resampling")
            try:
                waveform = torchaudio.functional.resample(waveform,orig_freq=samplerate,new_freq=self.samplerate)
            except:
                print("Could not resample!")
                print(path)
        if self.normalize:
            mabs = waveform.abs().max()
            if mabs > 1e-3:
                waveform /= mabs

        if self.remove_silence:
            waveform = librosa.effects.trim(waveform.numpy(),top_db=70)[0]
            waveform = torch.from_numpy(waveform)

        if self.augment_levels:
            waveform *= db2mag(torch.rand(1)*self.dbmin-self.dbmin)

        return waveform
                
    def __getitem__(self, n):
        L = np.random.randint(self.silence_samples)
        k = n
        output = torch.zeros(self.samples)
        flag = (L < self.samples)
        n_appended = 0
        while flag is True:
            waveform = self._load_audio(k)
            N = len(waveform)
            if (N+L) >= self.samples:
                N = self.samples - L
            starting = np.random.randint(self.starting_samples)
            ending = np.random.randint(self.ending_samples)

            if starting > N//2:
                starting = N//2
            N -= starting
            if ending > N//2:
                ending = N//2
            N -= ending

            output[L:L+N] = waveform[starting:starting+N]
            L += N
            S = np.random.randint(self.silence_samples)
            L += S
            k = np.random.randint(self.n_files)

            n_appended += 1
            if n_appended >= self.max_files:
                flag = False
            if L >= self.samples:
                flag = False
        
        return output, self.samplerate

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


class AuditoryStreams(Dataset):
    """AuditoryStreams
    """

    def __init__(self,root: str, audio_ext=".wav"):

        self._path = Path(root)
        self._audio_dir = self._path / 'mix'
        self._audio_ext = audio_ext

        self._info = pd.read_json(self._path / 'mix.json',orient='index')

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, str, str):
            ``(waveform, sample_rate, info)``
        """
        info = self._info.iloc[n]
        mix_filepath = self._audio_dir / info['filename']
        stems_filepath = self._audio_dir / info['stemname']
        audio_mix, _ = torchaudio.load(mix_filepath)
        audio_stems, _ = torchaudio.load(stems_filepath)
        dict_out = dict(mix=audio_mix,stems=audio_stems)
        for i in ['delta_time','delta_semitone','physics1','physics2']:
            dict_out[i] = info[i]
        return dict_out

    def __len__(self) -> int:
        return len(self._info)
    
class RWC(Dataset):
    """RWC test set, pre-mixed
    """

    def __init__(self,root: str, audio_ext=".wav"):

        self._path = Path(root)
        self._audio_dir = self._path / 'audio'
        self._audio_ext = audio_ext

        self._info = pd.read_json(self._path / 'mix.json',orient='index')

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, str, str):
            ``(waveform, sample_rate, info)``
        """
        info = self._info.iloc[n]
        mix_filepath = self._audio_dir / info['filename']
        stems_filepath = self._audio_dir / info['stemname']
        audio_mix, _ = torchaudio.load(mix_filepath)
        audio_stems, _ = torchaudio.load(stems_filepath)
        dict_out = dict(mix=audio_mix,stems=audio_stems)
        for i in ['instrument1','instrument2']:
            dict_out[i] = info[i]
        return dict_out

    def __len__(self) -> int:
        return len(self._info)

class NSYNTH(Dataset):
    """NSYNTH test set, pre-mixed
    """

    def __init__(self,root: str, audio_ext=".wav"):

        self._path = Path(root)
        self._audio_dir = self._path / 'audio'
        self._audio_ext = audio_ext

        self._info = pd.read_json(self._path / 'mix.json',orient='index')

    def __getitem__(self, n: int):
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            (Tensor, int, str, str, str):
            ``(waveform, sample_rate, info)``
        """
        info = self._info.iloc[n]
        mix_filepath = self._audio_dir / info['filename']
        stems_filepath = self._audio_dir / info['stemname']
        audio_mix, _ = torchaudio.load(mix_filepath)
        audio_stems, _ = torchaudio.load(stems_filepath)
        dict_out = dict(mix=audio_mix,stems=audio_stems)
        for i in info.keys():
            if i not in ['filename','stemname']:
                dict_out[i] = info[i]
        return dict_out

    def __len__(self) -> int:
        return len(self._info)
    
class ImagePipeline(torch.nn.Module):
    def __init__(
        self,
        n_src = 1,
        duration = 128,
        shift=0.0,
    ):
        super().__init__()
        self.n_src = n_src
        self.chain = Chain(size=duration)
        self.shift = Shift(shift_amount=shift)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        data = self.chain(image)
        if self.n_src > 1:
            n = data.shape[0]
            idx = np.arange(n) + 1
            if n > 1:
                idx[-1] = 0
            data = self.shift(data[idx]) + self.shift(data)
        else:
            data = self.shift(data)
        return data
    
class AudioPipeline(torch.nn.Module):
    def __init__(
        self,
        n_fft=512,
        n_time=None,
        hop_length=256,
        stretch_factor=0.8,
        compression=1.0,
        output_realimag=False,
    ):
        super().__init__()
        
        self.n_time = n_time
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=1,normalized=True, hop_length=hop_length)
        self.compression = compression
        self.output_realimag = output_realimag

        self.spec_aug = torch.nn.Sequential(
            torchaudio.transforms.TimeStretch(stretch_factor, fixed_rate=True),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
            torchaudio.transforms.TimeMasking(time_mask_param=10),
        )

        self.stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, power=None,normalized=True, hop_length=hop_length)
        self.inverse_stft = torchaudio.transforms.InverseSpectrogram(n_fft=n_fft, normalized=True, hop_length=hop_length)

    def crop(self,spec):
        if self.n_time is not None:
            T = spec.shape[-1]
            if self.n_time < T:
                spec = spec.transpose(0,-1)[:self.n_time].transpose(0,-1)
            else:
                padding = self.n_time - T
                padding = (padding//2, padding-padding//2)
                spec = F.pad(spec,padding)
        return spec

    def forward_complex(self, waveform: torch.Tensor) -> torch.Tensor:
        spec = self.stft(waveform)
        spec = self.crop(spec)
        spec = spec.abs() ** self.compression * torch.exp(1j*spec.angle())
        return spec

    def invert(self,stft):
        stft_linear = stft.abs() ** (1.0 / self.compression) * torch.exp(1j*stft.angle())
        return self.inverse_stft(stft_linear)
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.output_realimag:
            spec = self.forward_complex(waveform)
            spec = torch.stack([spec.real, spec.imag],1)
        else:
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
    
class Chain(object):
    def __init__(self,size) -> None:
        self.size = size
        
    def __call__(self,input):
        n = input.shape[0]
        
        k = self.size // input.shape[-1]
        m = n // k
        size_in = k*input.shape[-1]
        data = torch.zeros(m,input.shape[-2],self.size,dtype=input.dtype, layout=input.layout, device=input.device)
        for mm in range(m):
            idx = mm*k
            data[mm,:,:size_in] = torch.cat([input[i+idx] for i in range(k)],-1)
                            
        return data


class Mixer(torch.nn.Module):
    def __init__(
        self,
        n_sources=1,
        average=False,
    ):
        super(Mixer,self).__init__()
        self.n_sources = n_sources
        self.average = average

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.n_sources == 1:
            return waveform, waveform.unsqueeze(1)
        else:
            n_batch = waveform.shape[0]
            sources = []
            perms = torch.arange(n_batch)
            for i in range(self.n_sources):
                sources.append(waveform[perms])
                perms = torch.roll(perms,1)
            sources = torch.stack(sources,1)
            if self.average:
                sources /= sources.shape[1]
            mix = sources.sum(1)
            return mix, sources
        


#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_set = RWC(root="/Volumes/JulianSSD/data/RWC_test/")

    data = test_set.__getitem__(500)
    pipeline = AudioPipeline(n_fft=512,n_time=None,hop_length=128,compression=1)

    X = pipeline(data['mix'])
    S = pipeline(data['stems'])
    
    n = np.random.randint(X.shape[0])
    fig,ax = plt.subplots(3,figsize=(6,6))
    ax[0].imshow(mag2db(X[0]),aspect='auto',origin='lower',cmap='jet')
    ax[1].imshow(mag2db(S[0]),aspect='auto',origin='lower',cmap='jet')
    ax[2].imshow(mag2db(S[1]),aspect='auto',origin='lower',cmap='jet')
    
# %%
