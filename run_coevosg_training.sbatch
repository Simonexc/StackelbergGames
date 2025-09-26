#!/bin/bash

#SBATCH --job-name=flipit_coevosg_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurm_logs/stackelberg_coevosg_training_%j.out
#SBATCH --error=slurm_logs/stackelberg_coevosg_training_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=01153165@pw.edu.pl
#SBATCH --account=mandziuk-lab

date; hostname; pwd

export TMPDIR=/mnt/evafs/scratch/shared/strochimiak/tmp/$SLURM_JOB_ID
mkdir -p $TMPDIR

cd /home2/faculty/strochimiak/StackelbergGames/src
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python train_coevosg.py configs/run/flipit_simple_nn/test_full_training_flipit_000.yaml

rm -rf $TMPDIR

date
