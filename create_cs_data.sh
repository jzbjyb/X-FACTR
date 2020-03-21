#!/usr/bin/env bash


inp1=$1
inp2=$2

dir=$(dirname "${inp1}")

shuf $inp1 > $inp1.shuf
shuf $inp2 > $inp2.shuf

split -l $[ $(wc -l ${inp1}.shuf|cut -d" " -f1) * 80 / 100 ] ${inp1}.shuf
mv ${dir}/xaa ${inp1}.train
mv ${dir}/xab ${inp1}.test

rm ${dir}/xaa
rm ${dir}/xab

split -l $[ $(wc -l ${inp2}.shuf|cut -d" " -f1) * 80 / 100 ] ${inp2}.shuf
mv ${dir}/xaa ${inp2}.train
mv ${dir}/xab ${inp2}.test
