#!/bin/bash
#SBATCH --job-name="rdrstonemo"
#SBATCH --account=def-nereusvc
#SBATCH --mem-per-cpu=3900   #4 Gb
#SBATCH --time=2:00:00
#SBATCH --ntasks=12 # CPUs

python 7_wind_RDRS_toNEMOgrid_hourly.py
