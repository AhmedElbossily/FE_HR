#!/bin/tcsh

##Resource Request

#SBATCH --job-name=fc_1_HR
#SBATCH --partition=p2GPU40
#SBATCH --ntasks=1
#SBATCH --exclusive

##SBATCH --gpus-per-task=1 # number of gpus per task
##SBATCH --constraint=v100 # select node with v100 GPU

#SBATCH --time=72:00:00
#SBATCH --mail-user=ahmed.elbossily@hereon.de
#SBATCH --mail-type=END
#SBATCH --output=/gpfs/work/elbossil/fc_1_HR/C-out.txt 
#SBATCH --error=/gpfs/work/elbossil/fc_1_HR/C-error.txt

cd /gpfs/work/elbossil/fc_1_HR/build/
./Mfree
