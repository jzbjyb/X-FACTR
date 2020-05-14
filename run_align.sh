#!/usr/bin/env bash
#SBATCH --mem=10000
#SBATCH --gres=gpu:1
#SBATCH --time=0
#SBATCH --output=slurm_out/slurm-%j.out

train_file=$1
#test_file=$2
model_name=$2
output=$3
epoch=$4

warmup=0
block_size=256
batch_size=3
raw_prob=0
cs_mlm_probability=0.5
save_step=20000
keep_model=1
args="${@:4}"

python scripts/run_language_modeling.py \
	--train_data_file ${train_file} \
	--output_dir ${output} \
	--model_type bert \
	--line_by_line \
	--mlm \
	--mlm_probability 0.15 \
	--cs_mlm_probability ${cs_mlm_probability} \
	--raw_prob ${raw_prob} \
	--block_size $block_size \
	--num_train_epochs $epoch \
	--per_gpu_train_batch_size ${batch_size} \
	--per_gpu_eval_batch_size ${batch_size} \
	--warmup_steps ${warmup} \
	--logging_steps ${save_step} \
	--save_steps ${save_step} \
	--save_total_limit ${keep_model} \
	--do_train \
	--align \
	--pretrain_model_name ${model_name}
	#--evaluate_during_training \
	#--eval_data_file ${test_file} \
	#--do_eval \
	#${args}
