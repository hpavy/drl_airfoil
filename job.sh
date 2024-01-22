#!/bin/bash
#
#SBATCH --job-name=pbo_Savonius_DRL
#SBATCH --output=pbo_Savonius_DRL.txt
#SBATCH --partition=MAIN
#SBATCH --qos=calcul
#
#SBATCH --nodes 1
#SBATCH --ntasks 64
#SBATCH --ntasks-per-core 1
#SBATCH --threads-per-core 1
#SBATCH --time=2-00:00:00
#
module load cimlibxx/drl/pbo
module load cimlibxx/master
pbo savonius.json
