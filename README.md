# FEx

## Run on Heron Cluster (SLURM Manager)
- Remove the old project if exist 
```
rm -rf FEx
```
- Clone the project 
```
git clone https://github.com/AhmedElbossily/FEx.git
```
- Enter your username and token
- Change the directory to `FEx`
```
cd FEx
```
- Create a directory `build`
```
mkdir build
```
- Change the directory to `build`
```
cd build
```
- Load the following modules
```
module load compilers/cuda/11.0
module load compilers/gnu/9.4.0
module load applications/utils/cmake3.25.2
```
- Check if the above modules are loaded
```
module list
```
- Build and compile the code
```
cmake .. && make -j80
```
- Change to the parent directory
```
cd ..
```
- Submit the job 
```
sbatch run_SLURM.sh
```
- To see which jobs are currently running type
```
squeue
```
- If you want to abort your job already submitted to a partition, search for the `job-id`` by
using `squeue`` and abort the job with
```
scancel jobid
```

## expected outputs 
- You can monitor the percentage of completion in the `output.txt` file inside the `build` directory
```
cat output.txt
```
- The output files will be saved in the `result` folder inside the `build` directory 


## Run on EG cluster (PBS manager)
**Make sure that the following modules are loaded:**
- cuda
- gcc
- cmake
- shared modules
- exporting the gcc path
- exporting the cmake path

**I am using the following commands on my cluster to load the above modules**
```
module load cuda11.5/blas/11.5.1
module load cuda11.5/fft/11.5.1
module load cuda11.5/toolkit/11.5.1
module load gcc8/8.5.0
module load cmake-gcc8/3.21.3
module load shared
export PATH="/cm/shared/apps/gcc8/8.5.0/bin/:$PATH}"
export PATH="/cm/local/apps/cmake-gcc8/3.21.3/bin/:$PATH}"
module list
```
*Other versions may be installed on your cluster*

## Building Instructions
```
git clone https://github.com/AhmedElbossily/SPH_GPU_debugging.git.
cd SPH_GPU_debugging
mkdir build
cd build
cmake .. && make 
make 
```
*neglect any warning messages*

## Running Instructions 
- Open the `run.sh` file and modify based on your cluster.
- Run the project using command: `qsub ./run.sh`

