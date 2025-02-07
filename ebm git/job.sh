#!/bin/bash
#SBATCH -n 4
#SBATCH --ntasks-per-node=4
#SBATCH -p batch-AMD
#SBATCH --time=400:00:00

source ~/.bashrc
conda activate bib


python -u ebm_model.py
