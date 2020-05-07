#!/usr/bin/env bash

probe=$1
models=$2  # a list of models joined by ","
langs=$3  # a list of languages joined by ","
inp_dir=$4  # dir to save output
args="${@:5}"

IFS=','
read -ra MODELS <<< "${models}"
read -ra LANGS <<< "${langs}"

for m in "${MODELS[@]}"; do
    for l in "${LANGS[@]}"; do
        echo "==========" $m $l ${args} "=========="
        pred_dir=${inp_dir}/${m}__${l}/
        temp_file=$(mktemp)
        python scripts/ana.py --task multi_eval --probe $probe --model $m --lang $l --inp ${pred_dir} "${@:5}" &> $temp_file
        grep 'overall number' $temp_file
        grep 'overall acc' $temp_file
        rm $temp_file
    done
done
