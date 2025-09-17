#!/bin/bash

#SBATCH --job-name=stackelberg_sweep_flipit_gnn
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_logs/stackelberg_agents_flipit_gnn_%j.out
#SBATCH --error=slurm_logs/stackelberg_agents_flipit_gnn_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=01153165@pw.edu.pl
#SBATCH --account=mandziuk-lab

date; hostname; pwd

export TMPDIR=/mnt/evafs/scratch/shared/strochimiak/tmp/$SLURM_JOB_ID
mkdir -p $TMPDIR

cd /home2/faculty/strochimiak/StackelbergGames/src
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python attacker_sweep_agent.py 2ierzf2r &
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python attacker_sweep_agent.py 5apdfs1n &
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python attacker_sweep_agent.py q3affp2f &
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python attacker_sweep_agent.py x2zf817d &
wait
rm -rf $TMPDIR

date
