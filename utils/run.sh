#!/bin/bash
#SBATCH --account=def-YOURPI
#SBATCH --nodes=1             
#SBATCH --tasks-per-node=1   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=1  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=4G       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=00-01:00:00  # DD-HH:MM:SS

# module load python/3.8

DATADIR=data
# SOURCEDIR=.
# OUTPUTDIR=output/svae

# # Prepare virtualenv
# virtualenv --no-download $SLURM_TMPDIR/env
# source $SLURM_TMPDIR/env/bin/activate
# pip install --no-index -r $SOURCEDIR/requirements.txt

# echo "Transfering data"
# tar -xf $DATADIR/RWC-wav44.tar -C $SLURM_TMPDIR
# # DATADIR=$SLURM_TMPDIR
# echo "Finished transfer. Setting up multi processing"


# echo "Loading tensorboard and running Python script"
# # Start training, -u forces python to be unbuffered so it prints immediately
# python -u $SOURCEDIR/utils/resampler.py --data-path $SLURM_TMPDIR/RWC_dataset

# echo "tar and move"
# tar -C $SLURM_TMPDIR -cf $DATADIR/RWC_dataset.tar  RWC_dataset

# echo "Extracting tar"
# tar -xf $DATADIR/RWC_dataset.tar 

# cp -r -C $SLURM_TMPDIR RWC_dataset $DATADIR

tar -xf $DATADIR/RWC-wav44.tar $DATADIR