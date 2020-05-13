#!/usr/bin/env bash
#SBATCH --mem=30000
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out

set -e

probe=$1
model=$2  # model to use
lang=$3  # language to use
fact_file=$4  # fact file
facts=$5  # a list of facts joined by ","
out_dir=$6  # dir to save output
args="${@:7}"

mkdir -p ${out_dir}

IFS=','
read -ra FACTS <<< "${facts}"

i=0
while [ $i -lt ${#FACTS[*]} ]; do
    f=${FACTS[$i]}

    echo "==========" $f ${args} "=========="
    filename=${out_dir}/${f}.out
    pred_dir=${out_dir}/${f}/
    echo "python scripts/probe.py --probe ${probe} --model ${model} --lang ${lang} --facts ${fact_file}:${f} --pred_dir $pred_dir ${args} &> $filename" > $filename
    python scripts/probe.py --probe ${probe} --model ${model} --lang ${lang} --facts ${fact_file}:${f} --pred_dir $pred_dir "${@:7}" &>> $filename
    tail -n 1 $filename

    i=$(( $i + 1));
done
