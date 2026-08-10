#!/bin/bash
#SBATCH --job-name=bundleMLE
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=9:00:00
#SBATCH --output=%x_%j.out
#SBATCH --mail-user=dtm63837@uga.edu
#SBATCH --mail-type=END,FAIL

ml Python/3.12.3-GCCcore-13.3.0

pip install pandas
pip install numpy
pip install scipy
pip install seaborn
pip install matplotlib
pip install prettytable
pip install statsmodels

python LOV_bundle.py
