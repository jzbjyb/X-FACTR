#!/usr/bin/env bash

inp=$1
suffix=''
out=$2
lang=$3

: '
inp1=$1
inp2=$2

dir=$(dirname "${inp1}")

shuf $inp1 > $inp1.shuf
shuf $inp2 > $inp2.shuf

split -l $[ $(wc -l ${inp1}.shuf|cut -d" " -f1) * 80 / 100 ] ${inp1}.shuf ${dir}/x
mv ${dir}/xaa ${inp1}.train
mv ${dir}/xab ${inp1}.test

split -l $[ $(wc -l ${inp2}.shuf|cut -d" " -f1) * 80 / 100 ] ${inp2}.shuf ${dir}/x
mv ${dir}/xaa ${inp2}.train
mv ${dir}/xab ${inp2}.test
'

# down sample
python scripts/sling_prep.py --task cw_split --lang $lang --inp $inp --out $inp --down_sample 0.2
# down sample with balanced data
python scripts/sling_prep.py --task cw_split --lang $lang --inp $inp --out $inp --down_sample 0.2 --balance_lang

# raw
python scripts/ft.py --inp $inp --suffix $suffix --out $out --lang $lang
# cs
python scripts/ft.py --inp $inp --suffix $suffix --out $out --lang $lang --replace
# cs with random alias
python scripts/ft.py --inp $inp --suffix $suffix --out $out --lang $lang --replace --random_alias
