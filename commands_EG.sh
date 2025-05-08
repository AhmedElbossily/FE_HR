#!/bin/bash

set -e
trap 'echo "******* FAILED *******" 1>&2' ERR

echo 'Removing Old Repository'   
rm -rf SPH_GPU_FSE

echo 'Cloning FSE projecy' 
git clone https://github.com/AhmedElbossily/SPH_GPU_FSE.git

echo 'Changing directory to SPH_GPU_FSE'  
cd SPH_GPU_FSE

echo 'Creating build directory' 
mkdir build

echo 'Changing the directory to build' 
cd build

echo 'Load module cuda11.5' 
module load cuda11.5

echo 'building the project' 
cmake ..

echo 'compiling the project'
make -j88

echo 'run the job'
qsub ../run.sh

echo 'Job is submitted!! to the cluster'
echo '*** Lets see the job on the cluster'
