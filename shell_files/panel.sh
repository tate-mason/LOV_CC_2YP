#!/bin/bash
#SBATCH --job-name=panel_ops		# Job name (testBowtie2)
#SBATCH --partition=highmem_p		# Partition name (batch, highmem_p, or gpu_p)
#SBATCH --nodes=1			# Number of compute nodes for resources to be spread out over (increase only if using MPI enabled software)
#SBATCH --ntasks=1			# 1 task (process) for below commands
#SBATCH --cpus-per-task=1	 	# CPU core count per task, by default 1 CPU core per task
#SBATCH --mem=300G			# Memory per node (4GB); by default using M as unit
#SBATCH --time=6:00:00              	# Time limit hrs:min:sec or days-hours:minutes:seconds
#SBATCH --output=%x_%j.txt              # Standard output log, e.g., testBowtie2_12345.out
#SBATCH --mail-user=dtm63837@uga.edu    # Where to send mail
#SBATCH --mail-type=END,FAIL          	# Mail events (BEGIN, END, FAIL, ALL)

ml Python/3.12.3-GCCcore-13.3.0 # Load software module and run bowtie2 below

pip install -qqq -r /scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/shell_files/requirements.txt

set -e

git pull

python /scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/Code/dat_expo/panel_dat/panel_convert.py
python /scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/Code/dat_expo/panel_dat/panel_merge.py
python /scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/Code/dat_expo/panel_dat/debug_joins.py
python /scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/Code/dat_expo/panel_dat/panel_concat.py
