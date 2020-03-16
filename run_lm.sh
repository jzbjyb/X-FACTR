#!/usr/bin/env bash

train_file=data/cs/el_en_mtrex/train.txt
test_file=data/cs/el_en_mtrex/test.txt
output=$1
warmup=0
epoch=5
block_size=256
batch_size=4
cs_mlm_probability=0.5
save_step=100000
args="${@:2}"

python scripts/run_language_modeling.py \
	--train_data_file ${train_file} \
	--eval_data_file ${test_file} \
	--output_dir ${output} \
	--model_type bert \
	--line_by_line \
	--mlm \
	--mlm_probability 0.15 \
	--cs_mlm_probability ${cs_mlm_probability} \
	--block_size $block_size \
	--num_train_epochs $epoch \
	--per_gpu_train_batch_size ${batch_size} \
	--per_gpu_eval_batch_size ${batch_size} \
	--warmup_steps ${warmup} \
	--evaluate_during_training \
	--logging_steps ${save_step} \
	--save_steps ${save_step} \
	--do_train \
	--do_eval \
	${args}
