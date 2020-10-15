#!/bin/bash

#SBATCH --job-name=three-layer
#SBATCH --output=slurm-three-layer-%j.out
#SBATCH --error=slurm-three-layer-%j.err
#SBATCH --time=12:00:00
#SBARCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres="gpu:4" # GPUs per Node
#SBATCH --cpus-per-task=4

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1

module load spack/0.1
module load gcc/8.3.0 # to get libquadmath
module load julia/1.4.1
module load cuda/10.1.243
module load openmpi/3.1.4-pmi-cuda
module load py-matplotlib/3.1.1

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR/../examples/

NH=256
NZ=256
HOURS=96

CUDA_VISIBLE_DEVICES=0 unbuffer julia --project three_layer_constant_fluxes.jl --Nh $NH --Nz $NZ --buoyancy-flux 5e-8  --momentum-flux 0     --hours $HOURS 2>&1 | tee cooling.out           &
CUDA_VISIBLE_DEVICES=1 unbuffer julia --project three_layer_constant_fluxes.jl --Nh $NH --Nz $NZ --buoyancy-flux 0     --momentum-flux -2e-4 --hours $HOURS 2>&1 | tee wind.out              &
CUDA_VISIBLE_DEVICES=2 unbuffer julia --project three_layer_constant_fluxes.jl --Nh $NH --Nz $NZ --buoyancy-flux 2e-8  --momentum-flux -1e-4 --hours $HOURS 2>&1 | tee cooling_with_wind.out &
CUDA_VISIBLE_DEVICES=3 unbuffer julia --project three_layer_constant_fluxes.jl --Nh $NH --Nz $NZ --buoyancy-flux -2e-8 --momentum-flux -4e-4 --hours $HOURS 2>&1 | tee heating_with_wind.out

sleep 42480 # sleep for 11.8 hours
