#!/usr/bin/env bash

conda create -n mlama37 -y python=3.7 && conda activate mlama37
pip install -r requirements.txt
pip install http://www.jbox.dk/sling/sling-2.0.0-py3-none-linux_x86_64.whl
pushd ../
wget https://github.com/antonisa/unimorph_inflect.git
popd
