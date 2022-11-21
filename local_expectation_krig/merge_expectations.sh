#!/bin/bash
#SBATCH --partition=short-serial 
#SBATCH --job-name=merge_pkl
#SBATCH -o %j.out 
#SBATCH -e %j.err 
#SBATCH --mem=48G
#SBATCH --time=01:00:00
module load jaspy
python merge_expectations.py
