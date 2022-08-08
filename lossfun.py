import torch
from utils import utils

class FreqSignalLoss():
    def __init__(self, losstype='cmse_compressed', avg_batch=False, device=None, dtype=torch.float32, fs=16e3):
        self.nfft = 512
        self.win_len = 512
        self.awin = torch.hann_window(self.win_len, dtype=dtype).to(device)
        self.losstype = losstype
        self.complexWeight = 1.0
        self.compexp = .5

        self.fs = fs
        self.avg_batch = avg_batch
        self.hop_length = self.win_len // 2

        self.salt = 1e-12
        self.tau = 0.0
        self.ratio = False

    def __call__(self, estimate, target, seq_weight=None):
        L = min(estimate.shape[1], target.shape[1])
        if seq_weight is not None:
            # sequence-wise signal weighting
            estimate = seq_weight * estimate[:,:L]
            target = seq_weight * target[:,:L]

        Xhat = torch.stft(estimate, n_fft=self.nfft, window=self.awin, hop_length=self.hop_length, return_complex=True)
        X = torch.stft(target, n_fft=self.nfft, window=self.awin, hop_length=self.hop_length, return_complex=True)

        # return per sequence metrics
        if self.losstype == 'cmse_compressed':
            loss = self.cmse_compressed(Xhat,X,tau=self.tau,ratio=self.ratio)
        elif self.losstype == 'cmae':
            loss = self.cmae(Xhat, X)
        else:
            loss = self.csdr(estSpec=Xhat,targetSpec=X)

        if self.avg_batch:
            loss = torch.mean(loss)
        return loss


    def cmse_compressed(self, estSpec, targetSpec,tau=0,ratio=False):
        """
        complex MSE with compressed magnitudes
        Weighting between compressed complex and compressed magnitude MSE
        format: [batch, sources, ]
        """
        c = self.compexp
        # calc magnitudes
        estMag = torch.sqrt(torch.clamp_min(estSpec.real**2 + estSpec.imag**2, self.salt))
        targetMag = torch.sqrt(torch.clamp_min(targetSpec.real**2 + targetSpec.imag**2, self.salt))
        # phases
        estPhase = estSpec / estMag
        targetPhase = targetSpec / targetMag
        # compressed spectra
        estMag = estMag**c
        targetMag = targetMag**c
        estComp = estMag * estPhase
        targetComp = targetMag * targetPhase

        # average over time, frequency (keep batch dim!)
        
        
        magMSE = torch.mean((targetMag - estMag)**2, dim=(1,2)) + self.salt + self.tau
        complexMSE = torch.mean(torch.abs(targetComp - estComp)**2, dim=(1,2))  + self.salt + self.tau

        if ratio == True:
            magMSE /= torch.mean((targetMag)**2, dim=(1,2)) + self.salt + self.tau
            complexMSE /= torch.mean(torch.abs(targetComp)**2, dim=(1,2)) + self.salt + self.tau

        # magic weighting of magnitude and complex MSE
        mse = (1.-self.complexWeight)*magMSE + self.complexWeight*complexMSE

        log_mse = 10*torch.log10(mse)
        return log_mse

    def cmae(self, estSpec, targetSpec, c=0.3):
        """
        complex mean absolute error (L1)
        """
        # calc magnitudes
        estMag = torch.sqrt(torch.clamp_min(estSpec.real**2 + estSpec.imag**2, self.salt))
        targetMag = torch.sqrt(torch.clamp_min(targetSpec.real**2 + targetSpec.imag**2, self.salt))

        # average over time, frequency (keep batch dim!)
        magLoss = torch.mean(torch.abs(targetMag - estMag), dim=(1,2))
        complexLoss = torch.mean(torch.abs(targetSpec - estSpec), dim=(1,2))

        # magic weighting of magnitude and complex loss
        loss = (1.-self.complexWeight)*magLoss + self.complexWeight*complexLoss

        log_loss = 10*torch.log10(loss + self.salt)
        return log_loss

    def csdr(self, estSpec, targetSpec):
        """
        complex sdr
        """
        complexEnergy = torch.mean(torch.abs(targetSpec), dim=(1,2)) + self.salt
        complexLoss = torch.mean(torch.abs(targetSpec - estSpec), dim=(1,2)) + self.salt

        log_loss = -10*(torch.log10(complexEnergy) - torch.log10(complexLoss))
        return log_loss
    
    
