#!/bin/bash

#SBATCH --job-name=stackelberg_single_training
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_logs/stackelberg_agents_%j.out
#SBATCH --error=slurm_logs/stackelberg_agents_%j.err
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=01153165@pw.edu.pl
#SBATCH --account=mandziuk-lab

date; hostname; pwd

export TMPDIR=/mnt/evafs/scratch/shared/strochimiak/tmp/$SLURM_JOB_ID
mkdir -p $TMPDIR

cd /home2/faculty/strochimiak/StackelbergGames/src
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python train_single_stage.py configs/run/attacker_sweep_agent.py iogny23a &
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python train_single_stage.py configs/run/attacker_sweep_agent.py l96va9u0 &
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python train_single_stage.py configs/run/attacker_sweep_agent.py 7vpm204b &
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python train_single_stage.py configs/run/attacker_sweep_agent.py wfdrg1d0 &
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python train_single_stage.py configs/run/attacker_sweep_agent.py ia7c53e5 &

rm -rf $TMPDIR

date
