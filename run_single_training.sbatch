#!/bin/bash

#SBATCH --job-name=stackelberg_single_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_logs/stackelberg_single_training_%j.out
#SBATCH --error=slurm_logs/stackelberg_single_training_%j.err
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=01153165@pw.edu.pl

date; hostname; pwd

cd /home2/faculty/strochimiak/StackelbergGames/src
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python train_single_stage.py configs/run/test_single_training_poachers.yaml 1

date
