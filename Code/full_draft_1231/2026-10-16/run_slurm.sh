#!/usr/bin/env bash
# Submit from this folder or its parent:
#   sbatch 2026-10-16/run_slurm.sh --draws 128 --starts 3
# Add cluster-specific options (e.g. --account or --partition) before the script.
#SBATCH --job-name=flavor-comparison
#SBATCH --partition=highmem_p		# Partition name (batch, highmem_p, or gpu_p)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=flavor-comparison-%j.out
#SBATCH --error=flavor-comparison-%j.err

ml Python/3.12.3-GCCcore-13.3.0 # Load software module and run bowtie2 below

pip install -qqq -r /scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/shell_files/requirements.txt

set -e

git pull
python /scratch/dtm63837/Kilts_Panel/LOV_CC_2YP/Code/full_draft_1231/2026-10-16/run_slurm.sh
echo "Finished flavor comparison: job=$run_id"
