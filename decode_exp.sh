#!/usr/bin/env bash
#SBATCH --mem=30000
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out

#set -e

probe=$1
model=$2
lang=$3
out_dir=$4
inits=$5  # a list of models joined by ","
iters=$6  # a list of models joined by ","
num_mask=$7
max_iter=$8

mkdir -p ${out_dir}

IFS=','
read -ra INITS <<< "${inits}"
read -ra ITERS <<< "${iters}"

for init in "${INITS[@]}"; do
    for iter in "${ITERS[@]}"; do

        if [[ $init == all && $iter == none ]]; then
            my_max_iter=1
        else
            my_max_iter=$max_iter
        fi

        filename=${out_dir}/init_${init}_iter_${iter}.out
        pred_dir=${out_dir}/init_${init}_iter_${iter}

        python scripts/probe.py --probe $probe --model $model --lang $lang --num_mask $num_mask \
            --pred_dir $pred_dir --max_iter $my_max_iter --init_method $init --iter_method $iter &> $filename

        if [[ $init != all && $iter != none ]]; then
            filename=${out_dir}/init_${init}_iter_${iter}_reprob.out
            pred_dir=${out_dir}/init_${init}_iter_${iter}_reprob

            python scripts/probe.py --probe $probe --model $model --lang $lang --num_mask $num_mask \
                --pred_dir $pred_dir --max_iter $my_max_iter --init_method $init --iter_method $iter --reprob &> $filename
        fi
    done
done
