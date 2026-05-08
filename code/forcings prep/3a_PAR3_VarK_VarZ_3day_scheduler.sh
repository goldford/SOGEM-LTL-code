#!/bin/bash
#SBATCH --job-name="nemo to 3daily asc"
#SBATCH --account=def-nereusvc
#SBATCH --mem-per-cpu=8000   #
#SBATCH --time=02:00:00 # 
#SBATCH --ntasks=3 # CPUs

python 6_PAR3_VarK_VarZ_3day.py