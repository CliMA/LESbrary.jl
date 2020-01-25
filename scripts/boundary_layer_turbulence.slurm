#!/bin/bash

#SBATCH --time=24:00:00    # walltime
#SBATCH --ntasks=4         # number of processor cores (i.e. tasks)
#SBATCH --nodes=1          # number of nodes
#SBATCH --mem-per-cpu=16G  # memory per CPU core
#SBATCH --gres gpu:1       # Number of GPUs per node you need in you job

#SBATCH --mail-user=alir@mit.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

#SBATCH -J "Oceananigans"
#SBATCH -p any
#SBATCH -o slurm.%N.%j_%x.out
#SBATCH -e slurm.%N.%j_%x.err

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load openmpi/3.1.4 cuda/10.0

cd $HOME/LESbrary/
julia --project simulation/boundary_layer_turbulence_simple.jl
