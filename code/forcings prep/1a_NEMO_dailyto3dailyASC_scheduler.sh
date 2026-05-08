#!/bin/bash
#SBATCH --job-name="nemo to daily asc"
#SBATCH --account=def-nereusvc
#SBATCH --mem-per-cpu=8000   #3900 = 4 Gb; run sacct to see that 5428928K per var required 
#SBATCH --time=02:00:00 # 
#SBATCH --ntasks=3 # CPUs

python 1_NEMO_dailyNEMO_to_3dayASC_vserver.py