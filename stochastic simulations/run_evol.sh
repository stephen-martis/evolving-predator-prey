#!/bin/bash -l

#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --constraint=haswell

# conda environment required packages: numpy, math, scipy, pqdict, pickle, argparse, copy, random, time
module load python
conda activate bioinformatics
python nextreaction.py -m $1 -s $2 -l $3 -u $4
