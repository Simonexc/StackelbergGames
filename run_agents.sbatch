#!/bin/bash

#SBATCH --job-name=stackelberg_single_training
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_logs/stackelberg_agents_%j.out
#SBATCH --error=slurm_logs/stackelberg_agents_%j.err
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=01153165@pw.edu.pl
#SBATCH --account=mandziuk-lab

date; hostname; pwd

export TMPDIR=/mnt/evafs/scratch/shared/strochimiak/tmp/$SLURM_JOB_ID
mkdir -p $TMPDIR

cd /home2/faculty/strochimiak/StackelbergGames/src
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python attacker_sweep_agent.py 4e1l8pxv &
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python attacker_sweep_agent.py b4kskf94 &
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python attacker_sweep_agent.py jlorje1z &
wait
rm -rf $TMPDIR

date
