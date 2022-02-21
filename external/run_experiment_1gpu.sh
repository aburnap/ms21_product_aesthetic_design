#!/bin/bash -l

#SBATCH --job-name=ExampleClusterRun
#SBATCH --partition=gpu
#SBATCH --mem=96GB
#SBATCH --cpus-per-task=6
#SBATCH --mail-user=alex.burnap@yale.edu
#SBATCH --output=experiment_output.txt
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=2-00:00:00

# Get conda
#module load miniconda
#module load anaconda3-2020.07-gcc-9.3.0-myrjwlf
#eval "$(conda shell.bash hook)"

echo "SLURM_JOBID="$SLURM_JOBID
echo " "
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo " "
echo $PATH
echo " "
echo $PATH | grep aesthetics
echo " "
echo $(conda info --envs)

conda activate aesthetics

python experiment.py

