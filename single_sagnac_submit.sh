#!/bin/bash

#Initialize Modules
source /etc/profile

#Load Julia and MPI Modules

module load julia/1.7.3

module load mpi/openmpi-4.1.3

mpirun julia single_sagnac_parallel.jl
