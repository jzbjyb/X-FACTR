#!/usr/bin/env bash

export TORCH_HOME=~/zhengbaj3/torch_home/
export CUDA_VISIBLE_DEVICES=9
python scripts/probe.py $*
