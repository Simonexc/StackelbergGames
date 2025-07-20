#!/bin/bash

#SBATCH --job-name=stackelberg_single_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_logs/stackelberg_single_training_%j.out
#SBATCH --error=slurm_logs/stackelberg_single_training_%j.err
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=01153165@pw.edu.pl
#SBATCH --account=mandziuk-lab

date; hostname; pwd

export TMPDIR=/raid/strochimiak/tmp/$SLURM_JOB_ID
mkdir -p $TMPDIR

cd /home2/faculty/strochimiak/StackelbergGames/src
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python train_single_stage.py configs/run/test_single_training_poachers.yaml 1

rm -rf $TMPDIR

date
