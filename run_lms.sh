#!/usr/bin/env bash
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out

train_file1=$1
train_file2=$2
output1=$3
output2=$4
epoch=$5

./run_lm.sh $train_file1 $output1 $epoch
./run_lm.sh $train_file2 $output2 $epoch
