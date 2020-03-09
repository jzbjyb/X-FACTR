#!/usr/bin/env bash

#set -e

model=$1  # model to use
lang=$2  # language to use
fact_file=$3  # fact file
facts=$4  # a list of facts joined by ","
out_dir=$5  # dir to save output
args="${@:6}"

mkdir -p ${out_dir}

IFS=','
read -ra FACTS <<< "${facts}"

i=0
while [ $i -lt ${#FACTS[*]} ]; do
    f=${FACTS[$i]}

    echo "==========" $f ${args} "=========="
    filename=${out_dir}/${f}.out
    pred_dir=${out_dir}/${f}/
    echo "python scripts/probe.py --model ${model} --lang ${lang} --facts ${fact_file}:${f} --pred_dir $pred_dir ${args} &> $filename" > $filename
    python scripts/probe.py --model ${model} --lang ${lang} --facts ${fact_file}:${f} --pred_dir $pred_dir "${@:6}" &>> $filename
    tail -n 1 $filename

    i=$(( $i + 1));
done
