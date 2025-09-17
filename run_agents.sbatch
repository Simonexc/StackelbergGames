#!/bin/bash

#SBATCH --job-name=stackelberg_sweep_flipit
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_logs/stackelberg_agents_flipit_%j.out
#SBATCH --error=slurm_logs/stackelberg_agents_flipit_%j.err
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=01153165@pw.edu.pl
#SBATCH --account=mandziuk-lab

date; hostname; pwd

export TMPDIR=/mnt/evafs/scratch/shared/strochimiak/tmp/$SLURM_JOB_ID
mkdir -p $TMPDIR

cd /home2/faculty/strochimiak/StackelbergGames/src
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python attacker_sweep_agent.py rrspfenv &
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python attacker_sweep_agent.py m4xajie2 &
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python attacker_sweep_agent.py inyytjhb &
/home2/faculty/strochimiak/.pyenv/versions/stackelberg/bin/python attacker_sweep_agent.py pe72rgyt &
wait
rm -rf $TMPDIR

date
