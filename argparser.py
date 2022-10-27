import argparse

parser = argparse.ArgumentParser(description='DSAE Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
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
parser.add_argument('--mix-data', type=str, default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='fname',
                    help='name of the dataset')
parser.add_argument('--kernel-size', type=int, default=8, metavar='fname',
                    help='size of sliding kernel (in time)')
parser.add_argument('--overlap', type=int, default=2, metavar='fname',
                    help='size of sliding kernel (in time)')
parser.add_argument('--num-sources', type=int, default=2, metavar='fname',
                    help='number of sources in the model')
parser.add_argument('--latent-dim', type=int, default=64, metavar='fname',
                    help='dimension of latent sources')
parser.add_argument('--rnn-dim', type=int, default=256, metavar='fname',
                    help='dimension of rnn')
parser.add_argument('--data-directory', type=str, default='/Users/julian/Documents/Code/data/mnist', metavar='fname',
                    help='Folder containing the spectrogram data')
parser.add_argument('--net-type', type=str, default='pooled', metavar='fname',
                    help='name of the dataset')
parser.add_argument('--num-layers', type=int, default=2, metavar='fname',
                    help='number of sources in the model')
parser.add_argument('--output-directory', type=str, default='results', metavar='fname',
                    help='Folder containing the spectrogram data')