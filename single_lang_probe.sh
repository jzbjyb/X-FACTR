#!/usr/bin/env bash
#SBATCH --mem=30000
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out

#set -e

probe=$1
models=$2  # a list of models joined by ","
langs=$3  # a list of languages joined by ","
out_dir=$4  # dir to save output
args="${@:5}"

mkdir -p ${out_dir}

IFS=','
read -ra MODELS <<< "${models}"
read -ra LANGS <<< "${langs}"

i=0
while [ $i -lt ${#MODELS[*]} ]; do
    m=${MODELS[$i]}
    l=${LANGS[$i]}
    bm=$(basename "${m}")

    echo "==========" $m $l ${args} "=========="
    filename=${out_dir}/${bm}__${l}.out
    pred_dir=${out_dir}/${bm}__${l}/
    echo "python scripts/probe.py --probe $probe --model $m --lang $l --pred_dir $pred_dir ${args} &> $filename" > $filename
    python scripts/probe.py --probe $probe --model $m --lang $l --pred_dir $pred_dir "${@:5}" &>> $filename
    tail -n 1 $filename

    i=$(( $i + 1));
done
