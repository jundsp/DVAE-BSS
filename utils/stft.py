#%%
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Fold
import scipy.signal as sig


def biorwin(swin, N_hop):
	"""find bi-orthogonal analysis window to synthesis window for perfect reconstruction
    
    sebastian braun
    """
	L = len(swin)
	awin = np.zeros_like(swin)
	for k in range(0, N_hop):
		idx = range(k, L, N_hop)
		H = swin[idx]
		awin[idx] = np.linalg.pinv(H[:,np.newaxis])
	return awin


def get_win(N_win, N_hop):
	"""get best window for minimal reconstruction error
    
    sebastian braun 
    """
	hopfrac = float(N_hop) / float(N_win)

	if hopfrac < 0.15: # 1/16
		win = sig.windows.blackmanharris(N_win, sym=False)
	elif hopfrac < 0.2: # 1/8
		win = sig.windows.blackman(N_win, sym=False)
	elif hopfrac < 0.3: # 1/4
		win = sig.windows.hamming(N_win, sym=False)
	else:
		win = np.sqrt(sig.windows.hann(N_win, sym=False))

	return win


class STFT(torch.nn.Module):
    """non-trainable STFT layer"""

    def __init__(self, fs, winlen=.02, hopfrac=.5, nfft=None):
        super(STFT, self).__init__()
        """
        fs          sampling rate
        winlen      window length (s)
        hopfrac     hop fraction of window length
        nfft        FFT size (samples), optional
        """
        self.winlen = int(float(winlen)*fs)
        if nfft is not None:
            self.nfft = int(nfft)
        else:
            self.nfft = self.winlen

        self.N_fft = self.nfft
        self.hopsize = int(self.winlen * float(hopfrac))
        self.frameShift = self.hopsize / fs
        win = get_win(self.winlen, self.hopsize)
        self.register_buffer('win', torch.from_numpy(win).float())

        self.padder = Padder(fs, winlen=winlen, hopfrac=hopfrac, nfft=nfft)

    def forward(self, x):
        """
            x:  [batch x time] 
            X: [batch x frames x freq x 2]
        """

        xpad = self.padder(x)

        slices = torch.arange(self.winlen).view(1,-1) + torch.arange(self.padder.M).view(-1,1)*self.hopsize
        sig_slices = xpad[:,slices] * self.win.view(1,1,-1)
        X = torch.fft.rfft(sig_slices, n=self.nfft)
        X = torch.stack([X.real, X.imag],-1)

        return X


class iSTFT(torch.nn.Module):
    """non-trainable STFT layer"""

    def __init__(self, fs, winlen=.02, hopfrac=.5, nfft=None):
        super(iSTFT, self).__init__()
        """
        fs          sampling rate
        winlen      window length (s)
        hopfrac     hop fraction of window length
        nfft        FFT size (samples), optional
        """
        self.winlen = int(float(winlen)*fs)
        if nfft is not None:
            self.nfft = int(nfft)
        else:
            self.nfft = self.winlen

        self.N_fft = self.nfft
        self.hopsize = int(self.winlen * float(hopfrac))
        self.frameShift = self.hopsize / fs

        win = get_win(self.winlen, self.hopsize)
        win = biorwin(win, self.hopsize)
        self.register_buffer('win', torch.from_numpy(win).float())

    def forward(self, X, padder):
        """X: [batch x frames x freq x 2] """
        if len(X.shape) > 3:
            # complex input as channels
            Y = torch.complex(X[:,:,:,0], X[:,:,:,1]).contiguous()
        
        ypad_slices = torch.fft.irfft(Y,n=self.nfft)[:,:,:self.winlen] * self.win.view(1,1,-1)

        fold = Fold(output_size=(1,padder.L),kernel_size=(1,self.winlen),stride=self.hopsize)
        ypad = fold(ypad_slices.transpose(-2,-1)).view(-1,padder.L)

        y = padder.unpad(ypad)
        return y

class Padder(torch.nn.Module):
    """non-trainable padder module"""

    def __init__(self, fs, winlen=.02, hopfrac=.5, nfft=None):
        super(Padder, self).__init__()
        """
        fs          sampling rate
        winlen      window length (s)
        hopfrac     hop fraction of window length
        nfft        FFT size (samples), optional
        """
        self.winlen = int(float(winlen)*fs)
        if nfft is not None:
            self.nfft = int(nfft)
        else:
            self.nfft = self.winlen

        self.N_fft = self.nfft
        self.hopsize = int(self.winlen * float(hopfrac))
        self.frameShift = self.hopsize / fs

    def pad(self,x):
        self.siglen = x.shape[-1]

        self.padding = np.zeros(2).astype(int)
        self.padding[0] = self.winlen-self.hopsize

        self.M = np.ceil((self.winlen+self.siglen - 2.0*self.hopsize-1)/self.hopsize + 1).astype(int)
        self.padding[1] = ((self.M-1)*self.hopsize + self.winlen) - (self.padding[0] + self.siglen)

        xpad = F.pad(x,tuple(self.padding))
        self.L = xpad.shape[-1]
        return xpad

    def unpad(self,x):
        return x[:,self.padding[0]:-self.padding[1]]

    def forward(self, x, remove_pad=False):
        if remove_pad is True:
            return self.unpad(x)
        else:
            return self.pad(x)
        
def real2complex(x):
    return torch.complex(x[...,0], x[...,1]).contiguous()

def mag2db(x):
    return 20.0*torch.log10(torch.abs(x))

#%%
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import soundfile as sf
    audio, sr = sf.read("/Users/julian/Documents/MATLAB/sounds/greasy.wav")
    x = torch.from_numpy(audio).view(1,-1)

    kwds = dict(winlen=.032, hopfrac=.25, nfft=512)

    stft_layer = STFT(fs=16000,**kwds)
    istft_layer = iSTFT(fs=16000,**kwds)

    X = stft_layer(x)
    y = istft_layer(X,stft_layer.padder)

    plt.subplot(211)
    plt.plot(x[0]-y[0])
    plt.title('residual')
    plt.subplot(212)
    plt.imshow(mag2db(real2complex(X[0])).T,aspect='auto',origin='lower')
# %%
