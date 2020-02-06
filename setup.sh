#!/usr/bin/env bash

conda create -n lama37 -y python=3.7 && conda activate lama37
pip install -r requirements.txt
pushd ../
wget https://github.com/antonisa/unimorph_inflect.git
popd
