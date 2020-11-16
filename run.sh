#!/bin/bash
#SBATCH --account=def-depalle
#SBATCH --gres=gpu:t4:1  # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=64000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=01-00:00:00  # DD-HH:MM:SS
#SBATCH --mail-user=<you@example.com>
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

module load python/3.7
module load scipy-stack

DATADIR=~/scratch/data
SOURCEDIR=~/projects/def-depalle/jneri/ConvSVAE

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r $SOURCEDIR/requirements.txt

# Transfer the data
echo "Transferring the data"
# tar -xvf $DATADIR/MUMS/stft.tar -C $SLURM_TMPDIR/
tar -xvf $DATADIR/MNIST.tar -C $SLURM_TMPDIR/

echo "Loading tensorboard and running Python script"
# Start training
tensorboard --logdir=$SOURCEDIR/runs --host 0.0.0.0 &
python $SOURCEDIR/train.py \
--epochs 10000 \
--batch-size 64 \
--num-workers 8 \
--kernel-size 16 \
--overlap 2 \
--mix-data True \
--num-sources 5 \
--dataset MNIST \
--net-type pooled \
--num-layers 2 \
--data-directory $SLURM_TMPDIR/