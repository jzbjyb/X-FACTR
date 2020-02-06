#!/usr/bin/env bash

set -e

pip install -r requirements.txt
pip install http://www.jbox.dk/sling/sling-2.0.0-py3-none-linux_x86_64.whl
pushd ../
wget https://github.com/antonisa/unimorph_inflect.git
popd
