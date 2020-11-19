import torch
from torch import nn, optim
from torch.nn import functional as F
import math
import torchaudio
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import argparse
from torchvision import datasets, transforms
from model import DSAE
from dataset import *
from torchvision.utils import make_grid, save_image
from sklearn.manifold import  MDS
import seaborn as sns


parser = argparse.ArgumentParser(description='VAE TF Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--mix-data', type=str, default='True',
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='fname',
                    help='name of the dataset')
parser.add_argument('--kernel-size', type=int, default=8, metavar='fname',
                    help='size of sliding kernel (in time)')
parser.add_argument('--overlap', type=int, default=2, metavar='fname',
                    help='size of sliding kernel (in time)')
parser.add_argument('--num-sources', type=int, default=4, metavar='fname',
                    help='number of sources in the model')
parser.add_argument('--num-layers', type=int, default=2, metavar='fname',
                    help='number of sources in the model')
parser.add_argument('--data-directory', type=str, default='/Users/julian/Documents/Code/Compute_Canada/data', metavar='fname',
                    help='Folder containing the spectrogram data')
parser.add_argument('--net-type', type=str, default='pooled', metavar='fname',
                    help='name of the dataset')

jet = cm.get_cmap('jet', 256)
newcolors = jet(np.linspace(0, 1, 256))
newcolors[0] = (0.0,0.0,0.0,1.0)
mycmp = ListedColormap(newcolors)

def tolog(x):
    x = (x+1e-5).log()
    x -= x.min()
    x /= x.abs().max()
    return x

def process_data(data):
    if args.mix_data == 'True':
        n = data.size(0)//2
        data = 0.5*(data[:n] + data[n:n*2])
    return data.squeeze(1)

class ISTFT(nn.Module):
    def __init__(self, n_fft, hop_length = None):
        super(ISTFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft//4
        if self.hop_length is not None:
            self.hop_length = hop_length
            
        self.window = torch.hann_window(n_fft)

    def forward(self, x):
        x = torch.istft(x, n_fft = self.n_fft, hop_length=self.hop_length, win_length=None, window=self.window, center=True,  normalized=False, onesided=True)
        return x


def mp2comp(magnitude,phase):
    Zc = magnitude*torch.exp(1j*phase)
    Z = torch.view_as_real(Zc)
    return Z

def pad_full(Z):
    out = torch.zeros(Z.size(0),nfft//2+1,Z.size(2),2)
    out[:,:Z.size(1)] = Z
    return out

# Batch Synthesize
def synthesize(magnitude,phase):
    X = mp2comp(magnitude=magnitude,phase=phase)
    X = pad_full(X)* alpha 
    x = istft(X)
    return x

def dict_average(input):
    avg = {}
    for k,v in input.items():
        avg[k] = sum(v) / float(len(v))
    return avg

args = parser.parse_args()
torch.manual_seed(args.seed)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}

args.dataset = 'MNIST'
if args.dataset == 'MNIST':
    dim_data = (32,64)
    args.data_directory = '/Users/julian/Documents/Code/Compute_Canada/data'
    transform = transforms.Compose([transforms.CenterCrop(size=dim_data),transforms.RandomAffine(degrees=0,translate=(0.25,0.0)),transforms.ToTensor()])
    train_set = datasets.MNIST(args.data_directory, train=True, transform=transform)
    test_set = datasets.MNIST(args.data_directory, train=False, transform=transform)
if args.dataset == 'MUMS':
    dim_data = (128,512)
    args.data_directory = '/Volumes/Julian SSD/data/MUMS'
    train_set = MUMS(args.data_directory,train=True,duration=dim_data[1], bins=dim_data[0],magnitude=False)
    test_set = MUMS(args.data_directory, train=False, duration=dim_data[1], bins=dim_data[0],magnitude=False)
if args.dataset == 'Audio':
    dim_data = (128,64)
    # args.data_directory = '/Users/julian/Documents/Code/data/nsynth'
    train_set = Audioset(args.data_directory, spectrogram=True, duration=dim_data[1], bins=dim_data[0])
    test_set = Audioset(args.data_directory, spectrogram=True, duration=dim_data[1], bins=dim_data[0])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)

dimx = dim_data[0]
dimz = 16
dimh = 64
dimg = 64
model_name = ('saves/{}/model_{}_K{}_L{}_{}.pt').format(args.dataset,args.dataset,str(args.num_sources),str(args.num_layers),args.net_type)

model = DSAE(dimx=dimx,dimz=dimz,dimh=dimh,dimg=dimg,num_sources=args.num_sources,net_type=args.net_type,num_layers=args.num_layers)
model.load_state_dict(torch.load(model_name,map_location=torch.device('cpu')))
model.eval()

data = next(iter(test_loader))[0]
data = process_data(data.to(device))

if args.dataset == 'MUMS':
    stft = data
    data = torch.norm(data,2,-1)

recon_x, recon_s, mu_q, logvar_q, mu_p, logvar_p, mu0_q, logvar0_q = model(data)

m = 8
nrows = 2+args.num_sources
n = nrows*m
togrid = torch.zeros(n,1,data.size(-2),data.size(-1))
togrid[::nrows] = data[:m].unsqueeze(1)
togrid[1::nrows] = recon_x[:m].detach().unsqueeze(1)
for k in range(args.num_sources):
    togrid[2+k::nrows] = recon_s[:m,k].detach().unsqueeze(1)

sample, z , h = model.sample(M=8,N=mu_q.size(1)*2)
sample = sample.detach()

fig, ax = plt.subplots(8,2+args.num_sources)
k=0
for j in range(8):
	for i in range(2+args.num_sources):
		ax[j,i].imshow(togrid[k,0],vmin=0,vmax=1,cmap=mycmp),
		ax[j,i].axis('off')
		k+=1
ax[0,0].set_title('$x$')
ax[0,1].set_title('$\hat{x}$')
for i in range(args.num_sources):
	ax[0,2+i].set_title('$\hat{s}_'+str(i+1)+'$')
plt.savefig('results/'+str(args.dataset)+'/result_K'+str(args.num_sources)+'.pdf')
plt.show()


fig, ax = plt.subplots(8,args.num_sources,sharex=True, sharey=True)
for i in range(8):
    for j in range(args.num_sources):
    	ax[i,j].imshow(sample[i,j],vmin=0,vmax=1,cmap=mycmp),
    	ax[i,j].get_yaxis().set_visible(False)
for i in range(args.num_sources):
    ax[0,i].set_title('Sampled $s_'+str(i+1)+'$')
plt.savefig('results/'+str(args.dataset)+'/samples_K'+str(args.num_sources)+'.pdf')
plt.show()


znp = mu_q[:8].view(-1,dimz).detach().numpy()
mds = MDS(n_components=2)
mds_result = torch.from_numpy(mds.fit_transform(znp)).view(-1,mu_q.size(1),2)
for i in range(mds_result.size(0)):
    plt.plot(mds_result[i,:,0],mds_result[i,:,1])
    plt.scatter(mds_result[i,:,0],mds_result[i,:,1],alpha=.3)
plt.show()

for i in range(mds_result.size(0)):
    plt.plot(mds_result[i])
plt.show()

vaem = recon_s * (data / recon_x).unsqueeze(1)

if args.dataset == 'MUMS':
    print('Synthesizing Audio')
    fs = int(44100)
    nfft = 2048
    H = nfft // 4

    n = data.size(0)
    X = stft.squeeze(1)
    phase = torch.view_as_complex(X).angle()

    istft = ISTFT(n_fft=nfft,hop_length=H)
    alpha = np.sqrt(nfft)

    sound_gt = synthesize(magnitude=data.detach(),phase=phase)
    sound_vae = torch.zeros(sound_gt.size(0),args.num_sources,sound_gt.size(-1))
    sound_vaem = torch.zeros(sound_gt.size(0),args.num_sources,sound_gt.size(-1))
    for k in range(args.num_sources):
        sound_vae[:,k] = synthesize(magnitude=recon_s[:,k].detach(),phase=phase)
        sound_vaem[:,k] = synthesize(magnitude=vaem[:,k].detach(),phase=phase)
    sound_sample = synthesize(magnitude=sample.detach(),phase=sample.detach()*0)
    sound_sample /= sound_sample.abs().max()

    j = torch.randperm(sound_gt.size(0))[0]
    torchaudio.save(filepath='sounds/gt.wav',sample_rate=fs,src=sound_gt[j])
    for k in range(args.num_sources):
        torchaudio.save(filepath='sounds/estimate_'+str(k+1)+'.wav',sample_rate=fs,src=sound_vae[j,k])
        torchaudio.save(filepath='sounds/estimate_masked_'+str(k+1)+'.wav',sample_rate=fs,src=sound_vaem[j,k])
    torchaudio.save(filepath='sounds/sample.wav',sample_rate=fs,src=sound_sample[0])
