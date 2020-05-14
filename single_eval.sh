#!/usr/bin/env bash
#SBATCH --mem=20000
#SBATCH --cpus-per-task=8
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out

probe=$1
model=$2
lang=$3
inp_dir=$4  # dir to save output
args="${@:5}"

for f in $inp_dir/*/; do
    echo "==========" $f $args "=========="
    ls $f | wc -l
    temp_file=$(mktemp)
    python scripts/ana.py --task multi_eval --probe $probe --model $model --lang $lang --inp $f "${@:5}" &> $temp_file
    grep 'overall number' $temp_file
    grep 'overall acc' $temp_file
    rm $temp_file
done
