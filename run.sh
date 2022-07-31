#!/bin/bash
#SBATCH --account=def-depalle
#SBATCH --nodes=1  
#SBATCH --gres=gpu:1  # Request GPU "generic resources"
#SBATCH --cpus-per-task=8  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=00-08:00:00  # DD-HH:MM:SS

module load python/3.8
module load scipy-stack

DATADIR=~/scratch/data
SOURCEDIR=~/projects/def-depalle/jneri/GitHub/svae-bss

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r $SOURCEDIR/requirements.txt

# Transfer the data
echo "Transferring the data"
tar -xvf $DATADIR/MNIST.tar -C $SLURM_TMPDIR/

echo "Loading tensorboard and running Python script"
# Start training
tensorboard --logdir=$SOURCEDIR/runs --host 0.0.0.0 &
python $SOURCEDIR/train.py \
--epochs 10000 \
--batch-size 256 \
--num-workers 16 \
--kernel-size 16 \
--overlap 2 \
--mix-data True \
--num-sources 2 \
--dataset MNIST \
--net-type pooled \
--num-layers 2 \
--data-directory $SLURM_TMPDIR/