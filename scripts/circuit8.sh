#!/bin/bash

#SBATCH --job-name=circuit8
#SBATCH --output=/scratch/xgitiaux/circuit8_%j_%a.out
#SBATCH --error=/scratch/xgitiaux/circuit8_%j_%a.error
#SBATCH --mail-user=xgitiaux@gmu.edu
#SBATCH --mail-type=END
#SBATCH --export=ALL
#SBATCH --partition=all-LoPri
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --mem=64G
#SBATCH --qos=csqos
#SBATCH --array=0-0

module load python/3.6.7
module load cuda/9.2
source ../fvae-env/bin/activate

echo $SLURM_ARRAY_TASK_ID
../fvae-env/bin/python3 run.py
