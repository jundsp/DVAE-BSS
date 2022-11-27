import argparse

parser = argparse.ArgumentParser(description='DSAE Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 512)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--save-interval', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='number of batches to wait for printing')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--mix-data', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--conv2d', action='store_true', default=False)
parser.add_argument('--augment-levels', action='store_true', default=False)
parser.add_argument('--average-mix', action='store_true', default=False)
parser.add_argument('--remove-silence', action='store_true', default=False)
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
parser.add_argument('--latent-dim', type=int, default=32, metavar='fname',
                    help='dimension of latent sources')
parser.add_argument('--rnn-dim', type=int, default=128, metavar='fname',
                    help='dimension of rnn')
parser.add_argument('--duration', type=float, default=8, metavar='fname',
                    help='duration in seconds')
parser.add_argument('--n-time', type=int, default=513, metavar='N',
                    help='Time frames')
parser.add_argument('--data-directory', type=str, default='/Users/julian/Documents/Code/data', metavar='fname',
                    help='Folder containing the spectrogram data')
parser.add_argument('--output-directory', type=str, default='/Users/julian/Desktop', metavar='fname', help='Folder to save output data')
parser.add_argument('--model-directory', type=str, default='/Users/julian/Desktop', metavar='fname',help='Folder that has models.')
parser.add_argument('--net-type', type=str, default='pooled', metavar='fname',
                    help='name of the dataset')
parser.add_argument('--num-layers', type=int, default=1, metavar='fname',
                    help='number of sources in the model')
parser.add_argument('--job-id', type=str, default='local', metavar='fname',
                    help='Slurm job id')
parser.add_argument('--compression', type=float, default=1)
parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')
parser.add_argument('--max-files', type=int, default=100, metavar='fname',
                    help='max number of sounds to concatenate into single stream')
parser.add_argument('--num-targets', type=int, default=2, metavar='fname',
                    help='number of targets in the mixture concurrently (default 2)')
parser.add_argument('--cnn-channels',type=int,nargs='+',default=[32,64,128,128])
parser.add_argument('--net-type', type=str, default='svae', metavar='netname',
                    help='Network type, svae or vae')