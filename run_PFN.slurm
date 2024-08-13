#!/bin/bash
#SBATCH -A dune
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1

# Activate your virtual environment
source venv/bin/activate

python -u train_PFN.py 50