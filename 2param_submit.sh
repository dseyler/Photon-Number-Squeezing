#!/bin/bash

# Slurm sbatch options
#SBATCH -o 2param.sh.log-%j
#SBATCH -n 40
#SBATCH -N 1

#Initialize Modules
source /etc/profile

#Load Julia and MPI Modules

module load julia/1.7.3

module load mpi/openmpi-4.1.3

export SLURM_TASKS_PER_NODE=40

mpirun julia single_sagnac_2params.jl
