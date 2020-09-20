#!/usr/bin/env bash

set -e

echo 'sling'
pip install http://www.jbox.dk/sling/sling-2.0.0-py3-none-linux_x86_64.whl

echo 'unimorph inflect'
pushd ../
git clone https://github.com/antonisa/unimorph_inflect.git
popd

echo 'install kytea'
pushd ../
git clone https://github.com/neubig/kytea.git && cd kytea
autoreconf -i
./configure --prefix=$HOME/local
make && make install
popd

pip install -r requirements.txt

echo "transformer 2.4.1 has a bug with XLM-RoBERTa." \
    "Please replace line 114 in tokenization_xlm_roberta.py with" \
    "self.fairseq_tokens_to_ids[\"<mask>\"] = len(self.sp_model) + self.fairseq_offset"
