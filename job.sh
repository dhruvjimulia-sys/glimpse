#!/bin/bash
#PBS -lwalltime=00:00:30
#PBS -l select=1:ncpus=1:mem=2gb:ngpus=1
#PBS -N vis-chip-sim

cd ${PBS_O_WORKDIR}

# nvidia-smi

module purge
module load tools/prod
module load GCC/10.3.0
module load CUDA/12.2.0

make clean all
./build/main
