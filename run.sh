#!/bin/bash
#SBATCH --account=def-depalle
#SBATCH --nodes=1  
#SBATCH --gres=gpu:1  # Request GPU "generic resources"
#SBATCH --cpus-per-task=16  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=00-08:00:00  # DD-HH:MM:SS

module load python/3.8

DATADIR=~/scratch/data
SOURCEDIR=~/projects/def-depalle/jneri/GitHub/svae-bss

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index -r $SOURCEDIR/requirements.txt

echo "Loading tensorboard and running Python script"
# Start training
python $SOURCEDIR/model.py --data_directory /home/jneri/projects/def-depalle/jneri/data