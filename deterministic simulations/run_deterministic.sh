#!/bin/bash -l

#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --constraint=haswell

source activate bioinformatics
python deterministic.py -L $1 -v $2 -n $3