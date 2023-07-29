#!/bin/sh
#PBS -N experiment_run
#PBS -q gpu_1
#PBS -P CSCI1166
#PBS -l select=1:ncpus=10:mem=64gb
#PBS -l walltime=12:00:00
#PBS -o /mnt/lustre/users/rbieldt/cos_700/stdoutput.out
#PBS -e /mnt/lustre/users/rbieldt/cos_700/stderror.out
#PBS -m abe -M u13145992@tuks.co.za

ulimit -s unlimited
echo 'Starting run script'
cd /mnt/lustre/users/rbieldt/cos_700

module load chpc/python/anaconda/3-2021.11

source venv/bin/activate
cd src
runs="1"

for run in $runs
do
    echo 'Starting run: $run '
    python main.py $run
done
